#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import partial
from typing import Callable, Optional

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils._pytree import tree_map
from torch.utils.data import Dataset, DistributedSampler

from uni2ts.common import hydra_util  # noqa: hydra resolvers
from uni2ts.data.loader import DataLoader


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset | list[Dataset]],
    ):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = train_dataset

        if val_dataset is not None:
            self.val_dataset = val_dataset
            self.val_dataloader = self._val_dataloader

    @staticmethod
    def get_dataloader(
        dataset: Dataset,
        dataloader_func: Callable[..., DataLoader],
        shuffle: bool,
        world_size: int,
        batch_size: int,
        num_batches_per_epoch: Optional[int] = None,
    ) -> DataLoader:
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=None,
                rank=None,
                shuffle=shuffle,
                seed=0,
                drop_last=False,
            )
            if world_size > 1
            else None
        )
        return dataloader_func(
            dataset=dataset,
            shuffle=shuffle if sampler is None else None,
            sampler=sampler,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            self.train_dataset,
            instantiate(self.cfg.train_dataloader, _partial_=True),
            self.cfg.train_dataloader.shuffle,
            self.trainer.world_size,
            self.train_batch_size,
            num_batches_per_epoch=self.train_num_batches_per_epoch,
        )

    def _val_dataloader(self) -> DataLoader | list[DataLoader]:
        return tree_map(
            partial(
                self.get_dataloader,
                dataloader_func=instantiate(self.cfg.val_dataloader, _partial_=True),
                shuffle=self.cfg.val_dataloader.shuffle,
                world_size=self.trainer.world_size,
                batch_size=self.val_batch_size,
                num_batches_per_epoch=None,
            ),
            self.val_dataset,
        )

    @property
    def train_batch_size(self) -> int:
        return self.cfg.train_dataloader.batch_size // (
            self.trainer.world_size * self.trainer.accumulate_grad_batches
        )

    @property
    def val_batch_size(self) -> int:
        return self.cfg.val_dataloader.batch_size // (
            self.trainer.world_size * self.trainer.accumulate_grad_batches
        )

    @property
    def train_num_batches_per_epoch(self) -> int:
        return (
            self.cfg.train_dataloader.num_batches_per_epoch
            * self.trainer.accumulate_grad_batches
        )

# Transforms main function into a Hydra-managed entry point
# Sets Hydra version and the deafult config
# This config can import other configs, define default values, and use overrides
# from the cmd line
@hydra.main(version_base="1.3", config_name="default.yaml")
def main(cfg: DictConfig):
    # Optionally enable 32-bit floats
    if cfg.tf32:
        assert cfg.trainer.precision == 32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Instantiate lightning module using Hydra's instantiate function and 
    # convert all nested configs into Python primitives 
    model: L.LightningModule = instantiate(cfg.model, _convert_="all")

    # Optionally use torch.compile to use optimized kernels
    if cfg.compile:
        model.module.compile(mode=cfg.compile)
        
    # Instaitiate trainer 
    trainer: L.Trainer = instantiate(cfg.trainer)
    
    # Instantiate concat dataset builder and pass _args_ to ctor
    # Use the model's transformation map on the training datasets 
    train_dataset: Dataset = instantiate(cfg.data).load_dataset(
        model.train_transform_map
    )
    
    
    val_dataset: Optional[Dataset | list[Dataset]] = (
        tree_map(
            lambda ds: ds.load_dataset(model.val_transform_map),
            instantiate(cfg.val_data, _convert_="all"),
        )
        if "val_data" in cfg
        else None
    )
    L.seed_everything(cfg.seed + trainer.logger.version, workers=True)
    trainer.fit(
        model,
        datamodule=DataModule(cfg, train_dataset, val_dataset),
        ckpt_path=cfg.ckpt_path,
    )


if __name__ == "__main__":
    main()
