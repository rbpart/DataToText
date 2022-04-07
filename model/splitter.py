from argparse import ArgumentError
from torch.utils.data import random_split

from model.dataset import IDLDataset


class Splitter():
    def __init__(self, dataset: IDLDataset) -> None:
        self.dataset = dataset

    def train_valid_split(self, train_size = None, folds = None, augment_size = 0.7):
        if train_size is not None and folds is not None:
            raise ArgumentError("folds and train_size parameters can't be used at the same time")
        if train_size is not None:
            train, valid = random_split(self.dataset, [int(len(self.dataset)*train_size),int(len(self.dataset)*(1-train_size))+1])
            self.train_dataset = train
            self.train_dataset.indices = self.dataset.augment_data(train.indices, augment_size)
            self.valid_dataset = valid
            return train, valid
        if folds is not None:
            lengths = [int(len(self.dataset)/folds)]
            folds = random_split(self.dataset, lengths)
            self.folds = folds
            return folds

    # TODO : pass in the splitter
    # def folds(self, nfolds, save_every = 1000, clip = None, accumulate = 1):
    #     folds = self.train_valid_split(folds=nfolds)
    #     for i in range(folds):
    #         self.train_dataset = ConcatDataset(folds[0:i] + folds[i+1,nfolds])
    #         self.train_iterator = DataLoader(self.train_dataset, self.opts.batch_size,
    #                                 shuffle=True if not self.sampler else False,
    #                                 sampler = self.sampler, drop_last= True,
    #                                 pin_memory= True if self.opts.device == 'cuda' else False)
    #         self.valid_dataset = folds[i]
    #         self.valid_iterator = DataLoader(self.valid_dataset, self.opts.batch_size/2)
    #         self.num_batches = int(len(self.train_dataset)/self.opts.batch_size)
    #         self.train(save_every, clip = clip, accumulate = accumulate)
    #     return