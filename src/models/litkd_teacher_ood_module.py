from typing import Any, Tuple, List

import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch import nn

from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import Precision, Recall, F1Score, CohenKappa, ConfusionMatrix

from . import ood_algorithms
from ..domainbed import hparams_registry
from ..datamodules import ood_datasets
from ..domainbed.lib.query import Q
from ..domainbed.lib import reporting, misc
from ..domainbed import model_selection

import matplotlib
import numpy as np
from mlxtend.plotting import plot_confusion_matrix
import json, copy

class LitKDTeacherOODModule(LightningModule):
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
                 algorithm_name: str = None,
                 dataset_name: str = None,
                 task: str = None,
                 n_cls: int = None,
                 data_dir: str = None,
                 test_envs: int = None,
                 seed: int = None,
                 input_shape: tuple = None,
                 hparams_seed: int = 0,
                 trial_seed: int = 260981,
                 eval_method: Any = None,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler = None,
                 results_dir: str = None,
                 ):
        super().__init__()

        if hparams_seed == 0:
            hparams = hparams_registry.default_hparams(algorithm_name, 
                                                       dataset_name)
        # else:
        #     hparams = hparams_registry.random_hparams(algorithm_name, 
        #                                               dataset_name,
        #         misc.seed_hash(hparams_seed, trial_seed))
        # if hparams:
        #     self.hparams.update(json.loads(args.hparams))
        self.hparams.update(hparams)
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.test_envs = test_envs
        self.seed = seed
        self.input_shape = tuple(input_shape)
        algorithm_class = ood_algorithms.get_algorithm_class(algorithm_name=algorithm_name)
        self.algorithm = algorithm_class(self.input_shape,
                                    n_cls,
                                    self.test_envs,
                                    self.hparams)
        
        self.class_ids = ['%s' % i for i in range(n_cls)]

        self.metrics_train = MetricCollection({
            'acc': Accuracy(),
            'preccision': Precision(num_classes=n_cls, average='macro'),
            'recall': Recall(num_classes=n_cls, average='macro'),
            'F1Score': F1Score(num_classes=n_cls),
            'CohenKappa': CohenKappa(num_classes=n_cls),
        },)

        # TODO(anisio): May be there is a better way to get num_enviroments
        dataset = vars(ood_datasets)[dataset_name](data_dir,
                                               self.test_envs, 
                                               hparams)
        self.num_enviroments = len(dataset)

        # TODO(anisio) See it is necessary to implement eval_weigths
        # self.eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
        self.eval_loader_names = ['env{}_in'.format(i)
            for i in range(self.num_enviroments)]
        self.eval_loader_names += ['env{}_out'.format(i)
            for i in range(self.num_enviroments)]
        # self.eval_loader_names += ['env{}_uda'.format(i)
            # for i in range(self.num_enviroments)]

        self.val_loss_dict = nn.ModuleDict()
        self.val_acc_best_dict = nn.ModuleDict()
        self.metrics_val_dict = nn.ModuleDict()
        for name in self.eval_loader_names:
            self.metrics_val_dict[name] = MetricCollection({
            'acc': Accuracy(),
            'preccision': Precision(num_classes=n_cls, average='macro'),
            'recall': Recall(num_classes=n_cls, average='macro'),
            'F1Score': F1Score(num_classes=n_cls),
            'CohenKappa': CohenKappa(num_classes=n_cls),
        },)

            self.val_loss_dict[name] = MeanMetric()
            self.val_acc_best_dict[name] = MaxMetric()

        self.test_loss_dict = nn.ModuleDict()
        self.test_acc_best_dict = nn.ModuleDict()
        self.metrics_test_dict = nn.ModuleDict()
        for name in self.eval_loader_names:
            self.metrics_test_dict[name] = MetricCollection({
            'acc': Accuracy(),
            'preccision': Precision(num_classes=n_cls, average='macro'),
            'recall': Recall(num_classes=n_cls, average='macro'),
            'F1Score': F1Score(num_classes=n_cls),
            'CohenKappa': CohenKappa(num_classes=n_cls),
            # 'ConfusionMatrix': ConfusionMatrix(num_classes=n_cls)
        },)

            self.test_loss_dict[name] = MeanMetric()
            self.test_acc_best_dict[name] = MaxMetric()

        self.test_cmat = nn.ModuleDict()
        for name in self.eval_loader_names:
            self.test_cmat[name] = ConfusionMatrix(num_classes=n_cls)

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        # # for averaging loss across batches
        self.train_loss = MeanMetric()     

        self.eval_method = self.hparams.eval_method

        self.all_results = []

    def on_train_start(self) -> None:
        for name in self.eval_loader_names:
            self.val_acc_best_dict[name].reset()
        pass
        
    def training_step(self, batch: Any, batch_idx: int):
        result = self.algorithm.update(batch, loop='train')
        loss, preds, targets = result['loss'], result['preds'], result['targets']

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/metrics/', self.metrics_train(preds, targets), on_step=False, on_epoch=True, prog_bar=False)

        self.train_acc(preds, targets)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        results = {}
        results['epoch'] = self.current_epoch
        results['args'] = {'test_envs' : self.test_envs}

        tmp_r = {}
        for name in self.eval_loader_names:
            tmp_res = self.algorithm.update(batch[name], loop='val')
            loss, preds, targets = tmp_res['loss'], tmp_res['preds'], tmp_res['targets']
            val_metrics = self.metrics_val_dict[name](preds, targets)
            self.log('val/%s/metrics/' % (name), val_metrics, on_step=False, on_epoch=True, prog_bar=False)

            # update and log metrics
            self.val_loss_dict[name](loss)
            self.log("val/%s/loss" % (name), self.val_loss_dict[name], on_step=False, on_epoch=True, prog_bar=True)
        
            results[name+'_acc'] = val_metrics['acc'].detach().cpu()
            tmp_r[name+'_acc'] = val_metrics['acc'].detach().cpu().numpy().tolist()

        tmp_r['step'] = self.current_epoch
        tmp_r['args'] = {'test_envs' : list(self.test_envs)}
        tmp_r['args']['dataset'] = self.hparams['dataset_name']
        tmp_r['args']['algorithm'] = self.hparams['algorithm_name']
        tmp_r['args']['seed'] = self.hparams['seed']
        tmp_r['args']['hparams_seed'] = self.hparams['hparams_seed']
        tmp_r['args']['trial_seed'] = self.hparams['trial_seed']
        self.all_results.append(tmp_r)

        results.update({
                'hparams': self.hparams,
            })

        rr = Q([results])
        run_acc = self.eval_method.run_acc(rr)
        run_acc = run_acc
        self.log("val/acc", run_acc['val_acc'], on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}
    
    def validation_epoch_end(self, outputs: List[Any]):
        pass
        # results = {}
        # results['epoch'] = self.current_epoch
        # for name in self.eval_loader_names:
        #     val_metrics = self.metrics_val_dict[name].compute()
        #     acc = float(val_metrics['acc'])
        #     self.val_acc_best_dict[name](acc)  # update best so far val acc
            
        #     results[name+'acc'] = val_metrics['acc']

        #     # otherwise metric would be reset by lightning after each epoch
        #     self.log("val/%s/acc_bes" % (name), self.val_acc_best_dict[name].compute(), prog_bar=True)
        #     self.metrics_val_dict[name].reset()

    def test_step(self, batch: Any = None, batch_idx: int = None):
        pass
        results = {}
        results['epoch'] = self.current_epoch
        results['args'] = {'test_envs' : self.test_envs}
        for name in self.eval_loader_names:
            tmp_res = self.algorithm.update(batch[name], loop='val')
            loss, preds, targets = tmp_res['loss'], tmp_res['preds'], tmp_res['targets']
            test_metrics = self.metrics_test_dict[name](preds, targets)
            self.log('test/%s/metrics/' % (name), test_metrics, on_step=False, on_epoch=True, prog_bar=True)

            # update and log metrics
            self.test_loss_dict[name](loss)
            self.log("test/%s/loss" % (name), self.test_loss_dict[name], on_step=False, on_epoch=True, prog_bar=True)
        
            results[name+'_acc'] = test_metrics['acc'].detach().cpu()

            import wandb
            top_pred_ids = preds.cpu().numpy().argmax(axis=1)
            ground_truth_class_ids = targets.cpu().numpy()
            # class_ids = ['%s' % i for i in set([ground_truth_class_ids])]
            wandb.log({'conf_mat/%s' % name :
                      wandb.plot.confusion_matrix(probs=None,
                                                  preds=top_pred_ids, 
                                                  y_true=ground_truth_class_ids,
                                                  class_names=self.class_ids)}
                    )

        results.update({
                'hparams': self.hparams,
            })
        rr = Q([results])
        run_acc = self.eval_method.run_acc(rr)
        run_acc = run_acc
        self.log("test/acc", run_acc['val_acc'], on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def format_mean(self, data, latex):
        """Given a list of datapoints, return a string describing their mean and
        standard error"""
        if len(data) == 0:
            return None, None, "X"
        mean = np.mean(list(data))
        err = np.std(list(data) / np.sqrt(len(data)))
        if latex:
            return mean, err, "{:.1f} $\\pm$ {:.1f}".format(mean, err)
        else:
            return mean, err, "{:.1f} +/- {:.1f}".format(mean, err)

    def print_table(self, table, header_text, row_labels, col_labels, colwidth=10,
        latex=True):
        """Pretty-print a 2D array of data, optionally with row/col labels"""
        print("")

        if latex:
            num_cols = len(table[0])
            print("\\begin{center}")
            print("\\adjustbox{max width=\\textwidth}{%")
            print("\\begin{tabular}{l" + "c" * num_cols + "}")
            print("\\toprule")
        else:
            print("--------", header_text)

        for row, label in zip(table, row_labels):
            row.insert(0, label)

        if latex:
            col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}"
                for col_label in col_labels]
        table.insert(0, col_labels)

        for r, row in enumerate(table):
            misc.print_row(row, colwidth=colwidth, latex=latex)
            if latex and r == 0:
                print("\\midrule")
        if latex:
            print("\\bottomrule")
            print("\\end{tabular}}")
            print("\\end{center}")

    def test_epoch_end(self, outputs: List[Any]):
        results_file_name = '%s/results.jsonl' % (self.hparams['results_dir'])
        with open(results_file_name, 'w') as f:
            for r in self.all_results:
                f.write(json.dumps(r, sort_keys=True) + "\n")

        print(results_file_name)
        records = reporting.load_records(self.hparams['results_dir'])
        SELECTION_METHODS = [
            (model_selection.IIDAccuracySelectionMethod, 'IIDAccuracy'),
            (model_selection.LeaveOneOutSelectionMethod, 'LeaveOneOut'),
            (model_selection.OracleSelectionMethod, 'Oracle')
        ]

        for selection_method in SELECTION_METHODS:
            grouped_records = reporting.get_grouped_records(records).map(lambda group:
                { **group, "sweep_acc": selection_method[0].sweep_acc(group["records"]) }
            ).filter(lambda g: g["sweep_acc"] is not None)

            dataset = self.hparams['dataset_name']
            algorithm = self.hparams['algorithm_name']
            test_env = self.hparams['test_envs'][0]

            trial_accs = (grouped_records
                            .filter_equals(
                                "dataset, algorithm, test_env",
                                (dataset, algorithm, test_env)
                            ).select("sweep_acc"))
            mean, err, _ = self.format_mean(trial_accs, False)

            if mean != None and err != None:
                self.log("test/%s/acc_bes" % selection_method[1], mean, prog_bar=True)
                self.log("test/%s/acc_bes_err" % selection_method[1], err, prog_bar=True)
            elif mean == None:
                self.log("test/%s/acc_bes" % selection_method[1], -1.0, prog_bar=True)
            elif err == None:
                self.log("test/%s/acc_bes_err" % selection_method[1], -1.0, prog_bar=True)

            # print(mean, err)
            # print('---')


        # for selection_method in SELECTION_METHODS:
        #     grouped_records = reporting.get_grouped_records(records).map(lambda group:
        #         { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
        #     ).filter(lambda g: g["sweep_acc"] is not None)

        #     # read algorithm names and sort (predefined order)
        #     alg_names = Q(records).select("args.algorithm").unique()
        #     alg_names = ([n for n in ood_algorithms.ALGORITHMS if n in alg_names] +
        #         [n for n in alg_names if n not in ood_algorithms.ALGORITHMS])

        #     # read dataset names and sort (lexicographic order)
        #     dataset_names = Q(records).select("args.dataset").unique().sorted()
        #     dataset_names = [d for d in ood_datasets.DATASETS if d in dataset_names]
        #     for dataset in dataset_names:
        #         test_envs = range(ood_datasets.num_environments(dataset))
        #         table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        #         for i, algorithm in enumerate(alg_names):
        #             means = []
        #             for j, test_env in enumerate(test_envs):
        #                 trial_accs = (grouped_records
        #                     .filter_equals(
        #                         "dataset, algorithm, test_env",
        #                         (dataset, algorithm, test_env)
        #                     ).select("sweep_acc"))
        #                 mean, err, table[i][j] = self.format_mean(trial_accs, False)
        #                 means.append(mean)
        #             if None in means:
        #                 table[i][-1] = "X"
        #             else:
        #                 table[i][-1] = "{:.1f}".format(sum(means) / len(means))

        #         col_labels = [
        #             "Algorithm",
        #             *ood_datasets.get_dataset_class(dataset).ENVIRONMENTS,
        #             "Avg"
        #         ]
        #         header_text = (f"Dataset: {dataset}, "
        #             f"model selection method: {selection_method.name}")
        #         self.print_table(table, header_text, alg_names, list(col_labels),
        #             colwidth=20, latex=False)

        # results = {}
        # results['epoch'] = self.current_epoch
        # for name in self.eval_loader_names:
        #     test_metrics = self.metrics_test_dict[name].compute()
        #     acc = float(test_metrics['acc'])
        #     self.test_acc_best_dict[name](acc)  # update best so far val acc
            
        #     results[name+'acc'] = test_metrics['acc']

        #     # otherwise metric would be reset by lightning after each epoch
        #     self.log("test/%s/acc_bes" % (name), self.test_acc_best_dict[name].compute(), prog_bar=True)
        #     self.metrics_test_dict[name].reset()



            # # confusion matrix file and figure
            # cmat_tensor = self.test_cmat[name].compute()
            # cmat = cmat_tensor.cpu().numpy()
            # fig, ax = plot_confusion_matrix(
            #     conf_mat=cmat,
            #     # class_names=class_dict.values(),
            #     norm_colormap=matplotlib.colors.LogNorm()  
            #     # normed colormaps highlight the off-diagonals 
            #     # for high-accuracy models better
            # )
            # key = 'test/%s/confusion_matrix' % name
            # self.log(key, fig)
        
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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "litkd_teacher_ood.yaml")
    _ = hydra.utils.instantiate(cfg)