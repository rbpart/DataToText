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


class Trainer():
    def __init__(self, opts: HyperParameters = None, train_dataset: IDLDataset = None, valid_dataset: IDLDataset = None, test_dataset: IDLDataset = None,
                optim: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                criterion: nn.Module = None, model: DataToTextModel = None, checkpoint_file = None) -> None:
        if checkpoint_file is not None:
            self.checkpoint_file = checkpoint_file
        elif None not in [model,optim,scheduler]:
            self.model = model
            self.optim = optim
            self.scheduler = scheduler
        else:
            raise ValueError('Either provide a path to load checkpoint or provide all the required arguments to create a new model')

        if None not in [opts,train_dataset,valid_dataset, criterion]:
            self.opts = opts
            self.train_dataset = train_dataset
            self.train_iterator = DataLoader(train_dataset, opts.batch_size, shuffle=True)
            self.valid_dataset = valid_dataset
            self.valid_iterator = DataLoader(valid_dataset, opts.batch_size)
            self.num_batches = int(len(train_dataset)/self.opts.batch_size)
            self.criterion = criterion
        else:
            raise ValueError('Criterion and train and valid datasets should always be provided')

        if test_dataset is not None:
            self.test_dataset = test_dataset
            self.test_iterator = DataLoader(test_dataset, opts.batch_size)

    @property
    def avg_time(self):
        return np.mean(self.times)

    def train(self,save_every=1000):
        if self.checkpoint_file is not None:
            last_epoch = self.scheduler.state_dict()['last_epoch']
            print(f'Starting training with batch size {self.opts.batch_size} from checkpoint at epoch {last_epoch}:')
            for epoch in range(last_epoch, self.opts.num_epochs):
                self.times = []
                start_time= time.time()
                self.model.train()
                train_loss = self.train_one_epoch(epoch,save_every)
                self.model.eval()
                valid_loss = self.evaluate(self.valid_dataset,self.valid_iterator)
                self.report_epoch(epoch,start_time,train_loss,valid_loss)
        else:
            print(f'Starting training with batch size {self.opts.batch_size} from beginning')
            for epoch in range(self.opts.num_epochs):
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
            src, tar = self.train_dataset[batch]
            self.optim.zero_grad()
            out = self.model(src,tar)
            tar = tar.transpose(1,0)
            loss = 0
            for bn in range(len(batch)):
                loss += self.criterion(out[bn],tar[bn])
            loss.backward()
            self.optim.step()
            running_loss += loss.cpu().detach()
            self.report_iteration(epoch,i,self.scheduler.get_last_lr()[0],
                                loss.cpu().detach()/len(batch),(time.time()-running_time)/60)
            if epoch*self.opts.batch_size+i % save_every == save_every:
                self.save(epoch*self.opts.batch_size+i)
        self.scheduler.step()
        return running_loss


    def report_iteration(self,epoch,iter,lr,loss,time):
        self.times.append(time)
        print(f'| epoch {epoch:3d} | {iter:3d}/{int(self.num_batches)+1:5d} batches | '+
                  f'lr {lr:02.2f} | {time*1000:5.2f} ms/batch  | avg loss {loss:5.2f} | '+
                  f'ETA: {self.avg_time*(self.num_batches-iter):5.2f} min', end='\r')

    def report_epoch(self,epoch,start_time,train_loss,valid_loss):
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {(time.time()-start_time)/60:5.2f}s | '
                f'avg train loss {train_loss/len(self.train_dataset):5.2f} |avg valid loss {valid_loss/len(self.valid_dataset):5.2f} ')
            print('-' * 89)

    def evaluate(self, dataset, iterator):
        loss = 0
        for i, batch in tqdm(enumerate(iterator),total= int(len(dataset)/self.opts.batch_size)+1 ,desc='Loading batches...'):
            with torch.no_grad():
                src, tar = dataset[batch]
                out = self.model(src,tar)
                tar = tar.transpose(1,0)
                for bn in range(len(batch)):
                    loss += self.criterion(out[bn],tar[bn])
        return loss

    def save(self,num):
        os.mkdir(self.opts.save_path+f'checkpoint_{num}')
        torch.save(self.model.state_dict(),self.opts.save_path+f'model_{num}.pt')
        torch.save(self.optim.state_dict(),self.opts.save_path+f'optim_{num}.pt')
        torch.save(self.scheduler.state_dict(),self.opts.save_path+f'scheduler_{num}.pt')

    def load(self,checkpoint_file):
        num = checkpoint_file.split('_')[0]
        self.optim = torch.load('model/models/'+checkpoint_file+f'/optim_{num}.pt')
        self.model = torch.load('model/models/'+checkpoint_file+f'/model_{num}.pt')
        self.scheduler = torch.load('model/models/'+checkpoint_file+f'/scheduler_{num}.pt')