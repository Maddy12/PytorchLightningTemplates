# PytorchLightningTemplates
Consists of templates for running deep learning models in Pytorch Lightning.

All of these are modifications from Pytorch Lightning for the sake of keeping an organized experiment suite for my research. 
For assistance on use cases that are specific, see the [Pytorch Lightning Documentation](https://www.pytorchlightning.ai/)

## How to Use 
Step 1. Create a directory for your experiment.

Step 2. Create a `model.py` file where you will create your model.

Step 3. Set the root path for imports to the main directory. Example: 

```python
import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)

# If your model is two directories down from the main directory, otherwise just use `parent_dir`
top_level = os.path.dirname(parent_dir)
sys.path.insert(0, top_level)
```

Step 4. Import the base function and start creating a model object inheriting that function.

```python
from base import RunExperiment


class Model(RunExperiment):
    def __init__(self, hparams, **kwargs):
        """

        :param argparse.Namespace hparams:
        :param kwargs:
        """
        super(Model, self).__init__(hparams)
```

Step 5. Create a metric tracking function. 
Example: 
```python

class Accuracy(object):
    def __init__(self, split_model=False):
        self.correct = torch.tensor(0.)
        self.total = torch.tensor(0.)

    def reset(self):
        self.correct = torch.tensor(0.)
        self.total = torch.tensor(0.)

    def update(self, y_pred, y_true):
        y_pred_original = y_pred.clone()
        y_true_original = y_true.clone()

        # Accuracy
        correct = self.get_correct(y_true, y_pred)
        self.correct += correct
        self.total += y_true.shape[0]

        return correct / y_true.shape[0], y_pred_original, y_true_original

    def compute(self):
        acc = self.correct / self.total
        self.reset()
        return acc

    @staticmethod
    def get_correct(y_true, y_pred):
        if len(y_true.shape) > 1 and len(y_true.shape) < 3:
            # Find true label
            y_true = torch.argmax(y_true, dim=1)
        return float(torch.sum(torch.argmax(y_pred, 1) == y_true))
```

Step 6. Write a `init_metric_tracking` function in your model for the base class to call.
```python
# A metric tracking function that will generate a `self.train_accuracy` and `self.val_accuracy`.
 def init_metric_tracking(self):
        self.train_accuracy = YourMetricFunction(self.hparams.your_metric_args)
        self.val_accuracy = YourMetricFunction(self.hparams.your_metric_args)
```

Step 7. Write a `compute_loss` function in your model for the base class to call. Returning outputs and targets is useful if you have a list of outputs or targets because of a triplet loss or other reasons.
```python
def compute_loss(self, output, targets):
     # In case you have multiple outputs and targets, return what you want to be passed to your metrics update function.
     return YourLossFunction(output, targets), output, targets
```

Step 8. Write a `configure_optimizers` function in your model. Example:
```python
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=0.0000001)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
        return [optimizer], [scheduler]
```

Step 9. Implement specific arguments for your model by creating a `add_model_specific_args` function. Example:
```python
@staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Required
        parser.add_argument('--event_list_pth', type=str)
        parser.add_argument('--clips_dir', type=str)
        parser.add_argument('--dataset', default='kairos', type=str)
```

Step 10. Implement the remaining requirements, the same as Pytorch. Ex. ```def forward(self, inputs)```.

Step 11. Set up your script to run by initializing the `ArgumentParser` from the `argparse` package along with program level arguments. Finally 
Make sure you have imported `run_trainer` from the `utils/lightning_utils.py` to pass your model and parser to. 
```python
from utils.lightning_utils import run_trainer


if __name__ == "__main__":
    parser = ArgumentParser()

    # Program level args
    parser.add_argument('--checkpoint_pth', default=os.getcwd(), help='Where to store checkpoints', type=str)
    parser.add_argument('--resume', const=True, default=False, action="store_const")
    parser.add_argument('--test_only', const=True, default=False, action="store_const")
    parser.add_argument('--exp_name', default='inception_i3d', type=str)
    parser.add_argument('--version', type=str)
    parser.add_argument('--load_pth', default=None, type=str)
    parser.add_argument('--set_trace', const=True, default=False, action="store_const")

    # sys.settrace(trace)
    run_trainer(parser, Model)

```

Step 12. Run it by calling the model specific args and program specific args. Example:
The trainer level args can be found at the [PytorchLightning Documentation](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html).'

```bash
python3 experiments/nodewise_backprop/model.py --gpus $gpus \
                                               --exp_name $exp_name \
                                               --dropout_prob $dropout_prob \
                                               --version $version \
                                               --checkpoint_pth $checkpoint_pth \
                                               --batch_size $batch_size \
                                               --max_epochs $max_epochs \
                                               --accumulate_grad_batches $steps \
                                               --num_sanity_val_steps 1 
```

