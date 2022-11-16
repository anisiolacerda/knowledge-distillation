from typing import Any, List

import torch
from pytorch_lightning import LightningModule

from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import Precision, Recall, F1Score, CohenKappa, ConfusionMatrix

from models.components import model_dict

class LitKDTeacherIIDModule(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self,
                model_t_name: str = None,
                n_cls: int = None,
                optimizer: torch.optim.Optimizer = None,
                scheduler: torch.optim.lr_scheduler = None,
        ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.model = model_dict[model_t_name](num_classes=n_cls)
        self.criterion = torch.nn.CrossEntropyLoss()

        # # metric objects for calculating and averaging accuracy across batches
        n_cls = self.hparams.n_cls
        self.metrics_train = MetricCollection({
            'acc': Accuracy(),
            'preccision': Precision(num_classes=n_cls, average='macro'),
            'recall': Recall(num_classes=n_cls, average='macro'),
            'F1Score': F1Score(num_classes=n_cls),
            'CohenKappa': CohenKappa(num_classes=n_cls),
        },)

        self.metrics_val = MetricCollection({
            'acc': Accuracy(),
            'preccision': Precision(num_classes=n_cls, average='macro'),
            'recall': Recall(num_classes=n_cls, average='macro'),
            'F1Score': F1Score(num_classes=n_cls),
            'CohenKappa': CohenKappa(num_classes=n_cls),
        },)

        self.metrics_test = MetricCollection({
            'acc': Accuracy(),
            'preccision': Precision(num_classes=n_cls, average='macro'),
            'recall': Recall(num_classes=n_cls, average='macro'),
            'F1Score': F1Score(num_classes=n_cls),
            'CohenKappa': CohenKappa(num_classes=n_cls),
        },)

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def step(self, batch: Any):
        x, y = batch
        preds = self.model(x)
        loss = self.criterion(preds, y)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch=batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/metrics/', self.metrics_train(preds, targets), on_step=False, on_epoch=True, prog_bar=True)

        self.train_acc(preds, targets)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        # self.metrics_train.reset()
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        val_metrics = self.metrics_val(preds, targets)
        self.log('val/metrics/', val_metrics, on_step=False, on_epoch=True, prog_bar=True)

        # val_acc = float(val_metrics['acc'])
        
        self.val_acc(preds, targets)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        # self.log("val/acc", val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        val_metrics = self.metrics_val.compute()
        acc = float(val_metrics['acc'])
        self.val_acc_best(acc)  # update best so far val acc
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_bes", self.val_acc_best.compute(), prog_bar=True)
        # self.metrics_val.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/metrics/', self.metrics_test(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        
        self.test_acc(preds, targets)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        # self.metrics_test.reset()
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "litkd_teacher_iid.yaml")
    _ = hydra.utils.instantiate(cfg)
