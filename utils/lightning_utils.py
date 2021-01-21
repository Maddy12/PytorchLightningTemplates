import os
from typing import Optional, Any
import pandas as pd
import sys
import pdb
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn

import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


def run_trainer(parser, model):
    """
    Sample load_pth 'model_version=singlecrops_uniform_multicrops_dualbranch--epoch=0-val_acc=30.29-val_loss=6.26.ckpt'
    Required args in argparse:
    * checkpoint_pth (str): A directory to store model callbacks and Tensorboard logs
    * exp_name (str): The name of your overall experiment
    * version (str): The model version being run
    * gpus (int): Number of gpus available to run on
    * epochs (int): Number of epochs to train on.

    :parser namespace: The argsparse parser from the experiment Python file.
    :model pytorch_lightning.core.lightning.LightningModule: The model you are training.
    :return:
    """
    parser = Trainer.add_argparse_args(parser)
    parser = model.add_model_specific_args(parser)
    args = parser.parse_args()

    # If resuming a previous model or testing using a previous set of weights, load the model
    if args.resume or args.test_only:
        # assert 'load_pth' in vars(args.keys()), "If resuming or testing, please pass a checkpoint path to load weights."
        trainer = Trainer(resume_from_checkpoint=args.load_pth, gpus=args.gpus, max_epochs=args.max_epochs,
                          num_sanity_val_steps=1, weights_summary='full')

        # This loading will overwrite old hparams so that you can make minor changes or take into account new edits.
        if args.load_pth is None:
            model = model(args)
        else:
            model = model.load_from_checkpoint(args.load_pth, strict=True, **vars(args)).cuda()

        if args.test_only:
            trainer.test(model)
        else:
            trainer.fit(model)

    # Training from scratch
    else:
        # Ensure checkpoint path is existing
        checkpoint_pth = os.path.join(args.checkpoint_pth, args.exp_name)
        if not os.path.exists(checkpoint_pth):
            os.makedirs(checkpoint_pth)

        # Set up tensorboard logger with the checkpoint path and the name of your particular version of experiment.
        logger = TensorBoardLogger(save_dir=checkpoint_pth, name='tensorboard', version=args.version)

        weights_pth = os.path.join(checkpoint_pth, '{}_weights'.format(args.version),
                                   '{model_version}--{epoch}-{val_acc:.2f}-{val_loss:.2f}')
        model_checkpoint = ModelCheckpoint(filepath=weights_pth,
                                           monitor='val_acc', verbose=False, save_top_k=5,
                                           save_weights_only=False, mode='max', period=1)

        # Set up the trainer using the passed parameters, I use this way because the logger and model checkpoint.
        trainer = Trainer(checkpoint_callback=model_checkpoint, logger=logger, gpus=args.gpus,
                          max_epochs=args.max_epochs,
                          num_sanity_val_steps=args.num_sanity_val_steps,
                          accumulate_grad_batches=args.accumulate_grad_batches,
                          weights_summary='full',
                          distributed_backend=args.distributed_backend)

        # Create model object and train
        if args.set_trace:
            sys.settrace(trace)
        model = model(args)

        trainer.fit(model)


def _input_format_classification(preds: torch.Tensor, target: torch.Tensor, threshold: float):
    """ Convert preds and target tensors into label tensors
    Args:
        preds: either tensor with labels, tensor with probabilities/logits or
            multilabel tensor
        target: tensor with ground true labels
        threshold: float used for thresholding multilabel input
    Returns:
        preds: tensor with labels
        target: tensor with labels
    """
    if not (len(preds.shape) == len(target.shape) or len(preds.shape) == len(target.shape) + 1):
        raise ValueError(
            "preds and target must have same number of dimensions, or one additional dimension for preds"
        )

    if len(preds.shape) == len(target.shape) + 1:
        # multi class probabilites
        preds = torch.argmax(preds, dim=1)

    if len(preds.shape) == len(target.shape) and preds.dtype == torch.float:
        # binary or multilabel probablities
        preds = (preds >= threshold).long()
    return preds, target


def _confusion_matrix_update(preds: torch.Tensor,
                             target: torch.Tensor,
                             num_classes: int,
                             threshold: float = 0.5) -> torch.Tensor:
    preds, target = _input_format_classification(preds, target, threshold)
    unique_mapping = (target.view(-1) * num_classes + preds.view(-1)).to(torch.long)
    bins = torch.bincount(unique_mapping, minlength=num_classes ** 2)
    confmat = bins.reshape(num_classes, num_classes)
    return confmat


def _confusion_matrix_compute(confmat: torch.Tensor,
                              normalize: Optional[str] = None) -> torch.Tensor:
    allowed_normalize = ('true', 'pred', 'all', None)
    assert normalize in allowed_normalize, \
        f"Argument average needs to one of the following: {allowed_normalize}"
    confmat = confmat.float()
    if normalize is not None:
        if normalize == 'true':
            cm = confmat / confmat.sum(axis=1, keepdim=True)
        elif normalize == 'pred':
            cm = confmat / confmat.sum(axis=0, keepdim=True)
        elif normalize == 'all':
            cm = confmat / confmat.sum()
        nan_elements = cm[torch.isnan(cm)].nelement()
        if nan_elements != 0:
            cm[torch.isnan(cm)] = 0
            rank_zero_warn(f'{nan_elements} nan values found in confusion matrix have been replaced with zeros.')
        return cm
    return confmat


class ConfusionMatrix(Metric):
    """
        Computes the confusion matrix. Works with binary, multiclass, and multilabel data.
        Accepts logits from a model output or integer class values in prediction.
        Works with multi-dimensional preds and target.
        Forward accepts
        - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
        - ``target`` (long tensor): ``(N, ...)``
        If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument.
        This is the case for binary and multi-label logits.
        If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.
        Args:
            num_classes: Number of classes in the dataset.
            normalize: Normalization mode for confusion matrix. Choose from
                - ``None``: no normalization (default)
                - ``'true'``: normalization over the targets (most commonly used)
                - ``'pred'``: normalization over the predictions
                - ``'all'``: normalization over the whole matrix
            threshold:
                Threshold value for binary or multi-label logits. default: 0.5
            compute_on_step:
                Forward only calls ``update()`` and return None if this is set to False. default: True
            dist_sync_on_step:
                Synchronize metric state across processes at each ``forward()``
                before returning the value at the step. default: False
            process_group:
                Specify the process group on which synchronization is called. default: None (which selects the entire world)
        Example:
            >> target = torch.tensor([1, 1, 0, 0])
            >> preds = torch.tensor([0, 1, 0, 0])
            >> confmat = ConfusionMatrix(num_classes=2)
            >> confmat(preds, target)
            tensor([[2., 0.],
                    [1., 1.]])
    """

    def __init__(
            self,
            num_classes: int,
            normalize: Optional[str] = None,
            threshold: float = 0.5,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
    ):

        super().__init__(
            compute_on_step=compute_on_step,
            # dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.num_classes = num_classes
        self.normalize = normalize
        self.threshold = threshold

        allowed_normalize = ('true', 'pred', 'all', None)
        assert self.normalize in allowed_normalize, \
            f"Argument average needs to one of the following: {allowed_normalize}"

        self.add_state("confmat", default=torch.zeros(num_classes, num_classes),
                       dist_reduce_fx="sum")
        if not self.confmat.is_cuda:
            self.confmat = self.confmat.cuda(0)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        confmat = _confusion_matrix_update(preds, target, self.num_classes, self.threshold)
        if not confmat.is_cuda:
            confmat = confmat.cuda(0)
        self.confmat += confmat

    def compute(self) -> torch.Tensor:
        """
        Computes confusion matrix
        """
        return _confusion_matrix_compute(self.confmat, self.normalize)


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = self.one_hot(target, input.size(-1))
        logit = torch.nn.functional.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit)  # cross entropy
        loss = loss * (1 - logit) ** self.gamma  # focal loss

        return loss.sum()

    @staticmethod
    def one_hot(index, classes):
        size = index.size() + (classes,)
        view = index.size() + (1,)

        mask = torch.Tensor(*size).fill_(0).type_as(index)
        index = index.view(*view)
        ones = 1.

        if isinstance(index, torch.autograd.Variable):
            ones = torch.autograd.Variable(torch.Tensor(index.size()).fill_(1)).type_as(index)
            mask = torch.autograd.Variable(mask, volatile=index.volatile).type_as(index)

        return mask.scatter_(1, index, ones)


def trace(frame, event, arg):
    print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace


def update_state_dict(pth, model_state_dict):
    state_dict = torch.load(pth)
    missing = {k: v for k, v in model_state_dict.items() if k not in state_dict.keys()}
    state_dict.update(missing)
    return state_dict


def plot_confusion_matrix(pth, pretty_print=False, labels=None):
    cm = torch.load(pth).detach().cpu().numpy()
    if labels is not None:
        df_cm = pd.DataFrame(cm, index=[i for i in labels], columns=[i for i in labels])
    else:
        df_cm = pd.DataFrame(cm)
    if pretty_print:
        pretty_plot_confusion_matrix(df_cm, annot=True, figsize=[10, 10])
    else:
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm)
        plt.show()
    plt.close()


def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=8,
                                 lw=0.5, cbar=False, figsize=[12, 12], show_null_values=0, pred_val_axis='y'):
    """
      plot a pretty confusion matrix with seaborn
        Created on Mon Jun 25 14:17:37 2018
        @author: Wagner Cipriano - wagnerbhbr - gmail - CEFETMG / MMC
        REFerences:
          https://www.mathworks.com/help/nnet/ref/plotconfusion.html
          https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
          https://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python
          https://www.programcreek.com/python/example/96197/seaborn.heatmap
          https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/31720054
          http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py


      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if (pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    # this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    # thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    # set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal', fontsize=10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # iter in text elements
    array_df = np.array(df_cm.to_records(index=False).tolist())
    text_add = [];
    text_del = [];
    posi = -1  # from left to right, bottom to top.
    for t in ax.collections[0].axes.texts:  # ax.texts:
        pos = np.array(t.get_position()) - [0.5, 0.5]
        lin = int(pos[1]);
        col = int(pos[0]);
        posi += 1
        # print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        # set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # remove the old ones
    for item in text_del:
        item.remove()
    # append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    # titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  # set layout slim
    plt.show()


def get_new_fig(fn, figsize=[9, 9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()  # Get Current Axis
    ax1.cla()  # clear existing plot
    return fig1, ax1


def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = [];
    text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:, col]
    ccl = len(curr_column)

    # last line  and/or last column
    if (col == (ccl - 1)) or (lin == (ccl - 1)):
        # tots and percents
        if (cell_val != 0):
            if (col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif (col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif (lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%' % (per_ok), '100%'][per_ok == 100]

        # text to DEL
        text_del.append(oText)

        # text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d' % (cell_val), per_ok_s, '%.2f%%' % (per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy();
        dic['color'] = 'g';
        lis_kwa.append(dic);
        dic = text_kwargs.copy();
        dic['color'] = 'r';
        lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y - 0.3), (oText._x, oText._y), (oText._x, oText._y + 0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            # print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        # print '\n'

        # set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if (col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if (per > 0):
            txt = '%s\n%.2f%%' % (cell_val, per)
        else:
            if (show_null_values == 0):
                txt = ''
            elif (show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        # main diagonal
        if (col == lin):
            # set color of the textin the diagonal to white
            oText.set_color('w')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del


def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append(df_cm[c].sum())
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append(item_line[1].sum())
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col
    # print ('\ndf_cm:\n', df_cm, '\n\b\n')
