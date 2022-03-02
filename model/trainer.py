from torch import nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.dataset import IDLDataset
import torch
from model.parser import HyperParameters
from model.model import DataToTextModel
import time
import os
from torch.utils.data import random_split

class Trainer():
    def __init__(self, opts: HyperParameters = None, dataset: IDLDataset = None, test_dataset: IDLDataset = None,
                optim: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                criterion: nn.Module = None, model: DataToTextModel = None, checkpoint_file = None) -> None:

        self.checkpoint_file = checkpoint_file
        if checkpoint_file:
            self.load(checkpoint_file)
        elif None not in [model,optim,scheduler]:
            self.model = model
            self.optim = optim
            self.scheduler = scheduler
        else:
            raise ValueError('Either provide a path to load checkpoint or provide all the required arguments to create a new model')

        if None not in [opts,dataset, criterion]:
            self.opts = opts
            self.dataset = dataset
            self.criterion = criterion
            self.train_dataset = None
        else:
            raise ValueError('Criterion and train and valid datasets should always be provided')

        self.test_dataset = test_dataset
        if test_dataset:
            self.test_iterator = DataLoader(test_dataset, opts.batch_size)

    @property
    def avg_time(self):
        return np.mean(self.times)

    def train_k_fold(self, folds = 1, save_every = 1000):
        for fold in range(folds):
            self.train_valid_split(self.opts.train_size)
            self.train(save_every)
        return

    def train(self,save_every=1000):
        if self.train_dataset is None:
            self.train_valid_split(self.opts.train_size)
        if self.checkpoint_file:
            last_epoch = self.scheduler.state_dict()['last_epoch']
            print(f'Starting training with batch size {self.opts.batch_size} from checkpoint at epoch {last_epoch}:')
        else:
            last_epoch = 0
            print(f'Starting training with batch size {self.opts.batch_size} from beginning')

        for epoch in range(last_epoch, self.opts.num_epochs):
            self.times = []
            start_time= time.time()
            self.model.train()
            train_loss = self.train_one_epoch(epoch,save_every)
            self.model.eval()
            valid_loss = self.evaluate(self.valid_dataset,self.valid_iterator)
            self.report_epoch(epoch,start_time,train_loss,valid_loss)

        self.save('final')
        if self.test_dataset:
            self.evaluate(self.test_dataset,self.test_iterator)

    def train_one_epoch(self, epoch, save_every):
        running_loss = 0
        for i,batch in enumerate(self.train_iterator.batch_sampler,1):
            running_time = time.time()
            src, tar, lenghts, tar_tokens = self.train_dataset[batch]
            self.optim.zero_grad()
            out = self.model(src,tar,lenghts)
            tar = tar.transpose(1,0)
            loss = 0
            for bn in range(len(batch)):
                loss += self.criterion(out[bn],tar[bn])
            loss.backward()
            self.optim.step()
            running_loss += loss.cpu().detach()
            self.report_iteration(epoch,i,self.scheduler.get_last_lr()[0],
                                loss.cpu().detach()/len(batch),(time.time()-running_time)/60)
            if epoch*self.num_batches+i % save_every == 0:
                self.save(epoch*self.opts.batch_size+i)
        self.scheduler.step()
        return running_loss


    def report_iteration(self,epoch,iter,lr,loss,time):
        self.times.append(time)
        print(f'| epoch {epoch:3d} | {iter:3d}/{int(self.num_batches)+1:3d} batches | '+
                  f'lr {lr:1.5f} | {time*1000:5.2f} ms/batch  | avg loss {loss:5.2f} | '+
                  f'ETA: {self.avg_time*(self.num_batches-iter):5.2f} min', end='\r')

    def report_epoch(self,epoch,start_time,train_loss,valid_loss):
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | elapsed time: {(time.time()-start_time)/60:3.2f} min | '
                f'avg train loss {train_loss/len(self.train_dataset):5.2f} | avg valid loss {valid_loss/len(self.valid_dataset):5.2f} ')
            print('-' * 89)

    def evaluate(self, dataset, iterator: DataLoader):
        loss = 0
        for i, batch in tqdm(enumerate(iterator.batch_sampler),total= int(len(dataset)/self.opts.batch_size)+1 ,desc='Evaluation '):
            with torch.no_grad():
                src, tar, lenghts, tar_tokens = dataset[batch]
                out = self.model(src,tar)
                tar = tar.transpose(1,0)
                for bn in range(len(batch)):
                    loss += self.criterion(out[bn],tar[bn])
        return loss

    def train_valid_split(self, train_size):
        train, valid = random_split(self.dataset, [int(len(self.dataset)*train_size),int(len(self.dataset)*(1-train_size))+1])
        self.train_dataset = train
        self.train_iterator = DataLoader(self.train_dataset, self.opts.batch_size, shuffle=True ,
                                pin_memory= True if self.opts.device == 'cuda' else False)
        self.valid_dataset = valid
        self.valid_iterator = DataLoader(self.valid_dataset, self.opts.batch_size)
        self.num_batches = int(len(self.train_dataset)/self.opts.batch_size)
        return

    def save(self,num):
        checkpoint_file = self.opts.save_path+f'checkpoint_{num}/'
        os.mkdir(checkpoint_file)
        torch.save(self.model.state_dict(),checkpoint_file+f'model_{num}.pt')
        torch.save(self.optim.state_dict(),checkpoint_file+f'optim_{num}.pt')
        torch.save(self.scheduler.state_dict(),checkpoint_file+f'scheduler_{num}.pt')

    def load(self):
        num = self.checkpoint_file.split('_')[0]
        self.optim = torch.load('model/models/'+self.checkpoint_file+f'/optim_{num}.pt')
        self.model = torch.load('model/models/'+self.checkpoint_file+f'/model_{num}.pt')
        self.scheduler = torch.load('model/models/'+self.checkpoint_file+f'/scheduler_{num}.pt')