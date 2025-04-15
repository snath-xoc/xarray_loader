import copy
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
import sys

from .loading import get_IMERG_year
from .torch_batcher import BatchTruth
from .custom_collate_fn import CustomCollateFnGen

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_years=2018, val_years=2018, test_years=None,
        xbatch_size=[None,64,64], batch_size=8,
        train_epoch_size=1000, valid_epoch_size=200, test_epoch_size=1000, 
    ):
        super().__init__()
       
        self.datasets = {}
        self.batch_size=batch_size
        
        mult = 1
        if train_years is not None:
            
            df_truth = get_IMERG_year(train_years, months=2)
            self.datasets["train"] = BatchTruth(df_truth, batch_size=xbatch_size,antialiasing=True,
                                                  transform=transforms.RandomVerticalFlip(p=0.5),for_NJ=True,length=240)
                    
        if val_years is not None:

            df_truth = get_IMERG_year(val_years, months=2)
            self.datasets["valid"] = BatchTruth(df_truth, batch_size=[xbatch_size[0],300,300], weighted_sampler=False,
                                                      antialiasing=True, transform=transforms.RandomVerticalFlip(p=0.5),
                                                    for_NJ=True,length=240)
            
        if test_years is not None:
            df_truth = get_IMERG_year(val_years, months=2)
            self.datasets["test"] = BatchTruth(df_truth, batch_size=[xbatch_size[0],300,300], 
                                                      weighted_sampler=False, antialiasing=True,
                                                 for_NJ=True,length=240)
            
        else:
            self.datasets["test"] =  self.datasets["valid"]


    def dataloader(self, split):
        collate_fn, mult = CustomCollateFnGen(None)
        if split=="train":
            
            return DataLoader(
            self.datasets[split], batch_size=self.batch_size, collate_fn=collate_fn,
            pin_memory=True, num_workers=0, sampler=self.datasets[split].sampler,
            drop_last=True
            )
        else:
            return DataLoader(
                self.datasets[split],  collate_fn=collate_fn,
                pin_memory=True, num_workers=0, drop_last=True
            )

    def train_dataloader(self):
        return self.dataloader("train")

    def val_dataloader(self):
        return self.dataloader("valid")

    def test_dataloader(self):
        return self.dataloader("test")


