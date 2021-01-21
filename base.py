import warnings
import pdb
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
# from pytorch_lightning.metrics.classification.confusion_matrix import ConfusionMatrix

from argparse import ArgumentParser
from datetime import datetime
from multiprocessing import cpu_count
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)
from torch import optim
from torch.optim import lr_scheduler
import compress_pickle
import torch

# Local
from utils.lightning_utils import ConfusionMatrix

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class RunExperiment(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(RunExperiment, self).__init__()
        self.hparams = hparams
        self.save_pth = os.path.join(self.hparams.checkpoint_pth, self.hparams.exp_name)
        self.summarize_hparams()

        self.init_metric_tracking()
        self.train_conf_matrix = ConfusionMatrix(self.hparams.n_classes)
        self.val_conf_matrix = ConfusionMatrix(self.hparams.n_classes)

    def summarize_hparams(self):
        print("------------Start Hyperparameters------------")
        for param, val in self.hparams.items():
            print("{}: {}".format(param, val))
        print("-------------End Hyperparameters-------------")

    def hist_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        output = self.forward(x)

        loss, output, y_true = self.compute_loss(output, y_true)
        acc, output, y_true = self.train_accuracy.update(output, y_true)

        if len(y_true.shape) > 1:
            y_true = torch.argmax(y_true, dim=1)
        if len(output.shape) > 1:
            output = torch.argmax(output, dim=1)
        self.train_conf_matrix.update(output, y_true)

        if 'update_logs' in dir(self):
            self.update_logs(loss, acc, train=True)
        else:
            self.log('train_loss', loss, prog_bar=False, logger=True)
            self.log('train_acc', acc, prog_bar=False, logger=True)
        return loss

    def on_validation_epoch_start(self):
        self.hist_adder()

    def validation_step(self, batch, batch_idx):
        x, y_true = batch

        output = self.forward(x)
        loss, output, y_true = self.compute_loss(output, y_true)
        acc, output, y_true = self.val_accuracy.update(output, y_true)

        if len(y_true.shape) > 1:
            y_true = torch.argmax(y_true, dim=1)
        if len(output.shape) > 1:
            output = torch.argmax(output, dim=1)
        self.val_conf_matrix.update(output, y_true)

        if 'update_logs' in dir(self):
            self.update_logs(loss, acc, train=False)
        else:
            self.log('val_loss', loss, prog_bar=True, logger=True)
            self.log('val_acc', acc, prog_bar=True, logger=True)

        return loss

    def on_validation_epoch_end(self):
        train_acc = self.train_accuracy.compute()
        val_acc = self.val_accuracy.compute()

        train_conf = self.train_conf_matrix.compute()
        val_conf = self.val_conf_matrix.compute()

        if self.current_epoch % 10 == 0:
            torch.save(train_conf, os.path.join(self.save_pth, self.hparams.version + '_train_confusion_matrix.pt'))
            torch.save(val_conf, os.path.join(self.save_pth, self.hparams.version + '_val_confusion_matrix.pt'))

        self.log('epoch_train_acc', train_acc, logger=True, prog_bar=True)
        self.log('epoch_val_acc', val_acc, logger=True, prog_bar=True)
        print("\nEpoch {} -- Train {} -- Test {}\n".format(self.current_epoch, round(train_acc.item(), 2),
                                                           round(val_acc.item(), 2)))

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=False,
                                drop_last=True, num_workers=4, pin_memory=True, prefetch_factor=4)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False,
                                drop_last=True, num_workers=4, pin_memory=True, prefetch_factor=4)
        return dataloader

    def loss(self, logits, targets):
        return self.loss_func(logits, targets), logits
