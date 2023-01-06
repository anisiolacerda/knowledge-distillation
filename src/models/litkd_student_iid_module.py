from typing import Any, List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import Precision, Recall, AUC, F1Score, CohenKappa, ConfusionMatrix

from models.components import model_dict
from models.components.util import Embed, ConvReg, LinearEmbed
from models.components.util import Connector, Translator, Paraphraser

from .distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from .distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from .distiller_zoo.crd.criterion import CRDLoss

class DistillKL(torch.nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class LitKDStudentIIDModule(LightningModule):
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

    def __init__(
        #TODO(anisio) : descrive parameters
        self,
        model_t_path: str,
        model_s_name: str,
        n_cls: int,
        kd_T: float,
        gamma:float,
        alpha: float,
        beta: float,
        distill: str,
        #Specific (e.g. CC, CRD) parameters
        t_dim: int = None,
        s_dim: int = None,
        feat_dim: int = None,
        nce_k: int = None,

        nce_t: float = None,
        nce_m: float = None,
        n_data: int = None,
        
        hint_layer: float = 2,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model_t_path = model_t_path
        self.model_s = model_s_name
        self.n_cls = n_cls # number of classes
        self.kd_T = kd_T # temperature for KD distillation

        #TODO: investigate float attribution need
        self.gamma = gamma, # weight for classification
        self.gamma = float(self.gamma[0])    

        self.alpha = alpha, # weight balance for KD
        self.alpha = float(self.alpha[0])        

        self.beta = beta, # weight balance for other loesses
        self.beta = float(self.beta[0])        

        self.distill = distill
        self.hint_layer = hint_layer

        self.model_t_path = model_t_path
        self.model_t = self.load_teacher()
        self.model_s = model_dict[model_s_name](num_classes=self.n_cls)

        # loss function
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_div = DistillKL(self.kd_T)
        self.criterion_kd = self.get_criterion_kd()

        # # metric objects for calculating and averaging accuracy across batches
        self.metrics_train = MetricCollection({
            'acc': Accuracy(),
            'preccision': Precision(num_classes=self.n_cls, average='macro'),
            'recall': Recall(num_classes=self.n_cls, average='macro'),
            'F1Score': F1Score(num_classes=self.n_cls),
            'CohenKappa': CohenKappa(num_classes=self.n_cls),
        },)

        self.metrics_val = MetricCollection({
            'acc': Accuracy(),
            'preccision': Precision(num_classes=self.n_cls, average='macro'),
            'recall': Recall(num_classes=self.n_cls, average='macro'),
            'F1Score': F1Score(num_classes=self.n_cls),
            'CohenKappa': CohenKappa(num_classes=self.n_cls),
        },)

        self.metrics_test = MetricCollection({
            'acc': Accuracy(),
            'preccision': Precision(num_classes=self.n_cls, average='macro'),
            'recall': Recall(num_classes=self.n_cls, average='macro'),
            'F1Score': F1Score(num_classes=self.n_cls),
            'CohenKappa': CohenKappa(num_classes=self.n_cls),
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

    def get_rand_feat_s_t(self):
        data = torch.randn(2, 3, 32, 32)
        self.model_s.eval()
        self.model_t.eval()
        feat_t, _ = self.model_t(data, is_feat=True)
        feat_s, _ = self.model_s(data, is_feat=True)
        return feat_s, feat_t

    def get_criterion_kd(self):
        if self.distill == 'kd':
            criterion_kd = DistillKL(self.kd_T)
        elif self.distill == 'hint':
            criterion_kd = HintLoss()
            feat_s, feat_t = self.get_rand_feat_s_t()
            self.regress_s = ConvReg(feat_s[self.hint_layer].shape, feat_t[self.hint_layer].shape)
            # module_list.append(regress_s)
            # self.trainable_list.append(regress_s)
        elif self.distill == 'crd':
            feat_s, feat_t = self.get_rand_feat_s_t()
            self.hparams.s_dim = feat_s[-1].shape[1]
            self.hparams.t_dim = feat_t[-1].shape[1]
            # self.hparams.n_data = self.n_data
            criterion_kd = CRDLoss(self.hparams)
        #     module_list.append(criterion_kd.embed_s)
        #     module_list.append(criterion_kd.embed_t)
        #     trainable_list.append(criterion_kd.embed_s)
        #     trainable_list.append(criterion_kd.embed_t)
        elif self.distill == 'attention':
            criterion_kd = Attention()
        elif self.distill == 'nst':
            criterion_kd = NSTLoss()
        elif self.distill == 'similarity':
            criterion_kd = Similarity()
        elif self.distill == 'rkd':
            criterion_kd = RKDLoss()
        elif self.distill == 'pkt':
            criterion_kd = PKT()
        elif self.distill == 'kdsvd':
            criterion_kd = KDSVD()
        elif self.distill == 'correlation':
            feat_s, feat_t = self.get_rand_feat_s_t()
            criterion_kd = Correlation()
            self.embed_s = LinearEmbed(feat_s[-1].shape[1], self.feat_dim)
            self.embed_t = LinearEmbed(feat_t[-1].shape[1], self.feat_dim)
        #     module_list.append(embed_s)
        #     module_list.append(embed_t)
        #     trainable_list.append(embed_s)
        #     trainable_list.append(embed_t)
        elif self.distill == 'vid':
            feat_s, feat_t = self.get_rand_feat_s_t()
            s_n = [f.shape[1] for f in feat_s[1:-1]]
            t_n = [f.shape[1] for f in feat_t[1:-1]]
            criterion_kd = torch.nn.ModuleList(
                [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
            )
        #     # add this as some parameters in VIDLoss need to be updated
        #     trainable_list.append(criterion_kd)
        # elif self.distill == 'abound':
        #     s_shapes = [f.shape for f in feat_s[1:-1]]
        #     t_shapes = [f.shape for f in feat_t[1:-1]]
        #     connector = Connector(s_shapes, t_shapes)
        #     # init stage training
        #     init_trainable_list = torch.nn.ModuleList([])
        #     init_trainable_list.append(connector)
        #     init_trainable_list.append(self.model_s.get_feat_modules())
        #     criterion_kd = ABLoss(len(feat_s[1:-1]))
        #     init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, None, opt)
            # # classification
            # module_list.append(connector)
        # elif self.distill == 'factor':
        #     s_shape = feat_s[-2].shape
        #     t_shape = feat_t[-2].shape
        #     paraphraser = Paraphraser(t_shape)
        #     translator = Translator(s_shape, t_shape)
        #     # init stage training
        #     init_trainable_list = nn.ModuleList([])
        #     init_trainable_list.append(paraphraser)
        #     criterion_init = nn.MSELoss()
        #     init(model_s, model_t, init_trainable_list, criterion_init, train_loader, None, opt)
        #     # classification
        #     criterion_kd = FactorTransfer()
        #     module_list.append(translator)
        #     module_list.append(paraphraser)
        #     trainable_list.append(translator)
        # elif self.distill == 'fsp':
        #     s_shapes = [s.shape for s in feat_s[:-1]]
        #     t_shapes = [t.shape for t in feat_t[:-1]]
        #     criterion_kd = FSP(s_shapes, t_shapes)
        #     # init stage training
        #     init_trainable_list = torchnn.ModuleList([])
        #     init_trainable_list.append(model_s.get_feat_modules())
        #     init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, None, opt)
        #     # classification training
        #     pass
        else:
            raise NotImplementedError(self.distill) 

        return criterion_kd

    def get_loss_kd_value(self, feat_t, feat_s, index=None, contrast_idx=None):
        if self.distill == 'kd':
            loss_kd = 0
        elif self.distill == 'hint':
            # f_s = self.module_list[1](feat_s[self.hint_layer])
            f_s = self.regress_s(feat_s[self.hint_layer])
            f_t = feat_t[self.hint_layer]
            loss_kd = self.criterion_kd(f_s, f_t)
        elif self.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = self.criterion_kd(f_s, f_t, index, contrast_idx)
        elif self.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = self.criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif self.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = self.criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif self.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = self.criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif self.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = self.criterion_kd(f_s, f_t)
        elif self.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = self.criterion_kd(f_s, f_t)
        elif self.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = self.criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif self.distill == 'correlation':
            f_s = self.embed_s(feat_s[-1])
            f_t = self.embed_s(feat_t[-1])
            loss_kd = self.criterion_kd(f_s, f_t)
        elif self.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, self.criterion_kd)]
            loss_kd = sum(loss_group)
        elif self.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif self.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        # elif self.distill == 'factor':
        #     factor_s = module_list[1](feat_s[-2])
        #     factor_t = module_list[2](feat_t[-2], is_factor=True)
        #     loss_kd = self.criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(self.distill)

        return loss_kd

    def get_teacher_name(self):
        """parse teacher name"""
        segments = self.model_t_path.split('/')[-2].split('_')
        if segments[0] != 'wrn':
            return segments[0]
        else:
            return segments[0] + '_' + segments[1] + '_' + segments[2]

    def load_teacher(self):
        model_t = self.get_teacher_name()
        model = model_dict[model_t](num_classes=self.n_cls)
        model.load_state_dict(torch.load(self.model_t_path)['model'])
        return model

    def forward(self, x: torch.Tensor):
        return self.model_s(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()
        # self.val_acc_best_at5.reset()

    def step(self, batch: Any):
        index = None
        contrast_idx = None
        if self.hparams.distill == 'crd':
            x, y, index, contrast_idx = batch
        else:
            x, y = batch

        preact = False
        if self.distill in ['abound']:
            preact = True

        feat_s, logit_s = self.model_s(x, is_feat=True, preact=preact)
        feat_t, logit_t = self.model_t(x, is_feat=True, preact=preact)

        loss_cls = self.criterion_cls(logit_s, y) 
        loss_div = self.criterion_div(logit_s, logit_t)
        loss_kd = self.get_loss_kd_value(feat_t, feat_s, index, contrast_idx)

        loss = (self.gamma * loss_cls) + (self.alpha * loss_div) + (self.beta * loss_kd)

        preds = torch.argmax(logit_s, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "litkd.yaml")
    _ = hydra.utils.instantiate(cfg)
