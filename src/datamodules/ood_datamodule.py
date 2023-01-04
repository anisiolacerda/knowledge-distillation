from typing import Any, Dict, Optional, Tuple, List

from pytorch_lightning import LightningDataModule

from . import ood_datasets
from domainbed.lib import misc
from domainbed import hparams_registry
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

from pytorch_lightning.trainer.supporters import CombinedLoader

from copy import copy

class OODDataModule(LightningDataModule):
    def __init__(self,
                 algorithm_name: str = None,
                 dataset_name: str = None,
                 data_dir: str = './data/',
                 task: str = None,
                 n_cls: int = None,
                 test_envs: List = None,
                 input_shape: Tuple = None,
                 holdout_fraction: float = None,
                 uda_holdout_fraction: float = None,
                 hparams_seed: int = 0,
                 batch_size: int = None,
                 num_workers: int = None,
                 pin_memory: bool = None,
        ) -> None:        
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

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.test_envs = test_envs
        self.holdout_fraction = holdout_fraction
        self.uda_holdout_fraction = uda_holdout_fraction
        self.task = task

        if self.dataset_name in vars(ood_datasets):
            self.dataset = vars(ood_datasets)[self.dataset_name](self.data_dir,
                                                            self.test_envs, 
                                                            self.hparams)
        else:
            raise NotImplementedError

    def prepare_data(self) -> None:
        # TODO(anisio): include data download
        pass

    def setup(self, stage: Optional[str] = None):
        in_splits = []
        out_splits = []
        uda_splits = []
        for env_i, env in enumerate(self.dataset):
            uda = []

            out, in_ = misc.split_dataset(env, int(len(env)*self.holdout_fraction))
            # ,misc.seed_hash(args.trial_seed, env_i))

            if env_i in self.test_envs:
                uda, in_ = misc.split_dataset(in_, int(len(in_)*self.uda_holdout_fraction))
                    # ,misc.seed_hash(args.trial_seed, env_i))
    
            if self.hparams['class_balanced']:
                in_weights = misc.make_weights_for_balanced_classes(in_)
                out_weights = misc.make_weights_for_balanced_classes(out)
                if uda is not None:
                    uda_weights = misc.make_weights_for_balanced_classes(uda)
            else:
                in_weights, out_weights, uda_weights = None, None, None
            in_splits.append((in_, in_weights))
            out_splits.append((out, out_weights))
            if len(uda):
                uda_splits.append((uda, uda_weights))

        if self.task == "domain_adaptation" and len(uda_splits) == 0:
            raise ValueError("Not enough unlabeled samples for domain adaptation.")

        train_loaders = [InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=self.hparams['batch_size'],
            num_workers=self.dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(in_splits)
            if i not in self.test_envs]

        uda_loaders = [InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=self.hparams['batch_size'],
            num_workers=self.dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(uda_splits)]

        self.eval_loaders = [FastDataLoader(
            dataset=env,
            batch_size=64,
            num_workers=self.dataset.N_WORKERS)
            for env, _ in (in_splits + out_splits + uda_splits)]
            
        # self.eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
        self.eval_loader_names = ['env{}_in'.format(i)
            for i in range(len(in_splits))]
        self.eval_loader_names += ['env{}_out'.format(i)
            for i in range(len(out_splits))]
        # self.eval_loader_names += ['env{}_uda'.format(i)
            # for i in range(len(uda_splits))]
        
        self.train_minibatches_iterator = zip(*train_loaders)
        if self.task != "domain_adaptation":
            self.uda_minibatches_iterator = None
        self.uda_minibatches_iterator = zip(*uda_loaders)

        # self.eval_minibatches_iterator = zip(*self.eval_loaders)

    def train_dataloader(self):
        return self.train_minibatches_iterator

    def val_dataloader(self):
        loaders = {}
        evals = zip(self.eval_loader_names, self.eval_loaders)
        for name, ldr in evals:
            loaders[name] = ldr
        self.combined_loaders = CombinedLoader(loaders=loaders)

        self.test_combined_loaders = copy(self.combined_loaders)
        return self.combined_loaders

    def test_dataloader(self):
        # loaders = {}
        # evals = zip(self.eval_loader_names, self.eval_loaders)
        # for name, ldr in evals:
        #     loaders[name] = ldr
        # combined_loaders = CombinedLoader(loaders=loaders)

        return self.test_combined_loaders

    # def teardown(self, stage: str):
    #     # Used to clean-up when the run is finished
    #     ...
