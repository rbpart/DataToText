from torch import nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.dataset import IDLDataset
import torch
from model.parser import Opts
from model.model import DataToTextModel
import time
import os


class Trainer():
    def __init__(self, opts: Opts = None, train_dataset: IDLDataset = None, train_iterator: DataLoader = None,
                valid_dataset: IDLDataset = None, valid_iterator: DataLoader = None,optim: torch.optim.Optimizer = None,
                scheduler: torch.optim.lr_scheduler._LRScheduler = None, criterion: nn.Module = None, model: DataToTextModel = None,
                from_dir_path = None) -> None:
        if from_dir_path is not None:
            self.saved_path = from_dir_path
        elif None not in [opts,train_dataset,train_iterator,valid_dataset,valid_iterator,model,optim,scheduler,criterion]:
            self.saved_path = None
            self.opts = opts
            self.train_dataset = train_dataset
            self.train_iterator = train_iterator
            self.num_batches = len(train_dataset)/self.opts.batch_size
            self.valid_dataset = valid_dataset
            self.valid_iterator = valid_iterator
            self.model = model
            self.optim = optim
            self.scheduler = scheduler
            self.criterion = criterion
        else:
            raise ValueError('Either provide a path to load checkpoint or provide all the required arguments to create a new model')

    @property
    def avg_time(self):
        return np.mean(self.times)

    def train(self,save_every=1000):
        if self.saved_path is not None:
            last_epoch = self.scheduler.state_dict()['last_epoch']
            print(f'Starting training with batch size {self.opts.batch_size} from checkpoint at epoch {last_epoch}:')
            for epoch in range(last_epoch, self.opts.num_epochs):
                self.times = []
                startt= time.time()
                self.model.train()
                loss = self.train_one_epoch(epoch,save_every)
                self.model.eval()
                valid_loss = self.evaluate()
                print('-' * 89)
                print(f'| end of epoch {epoch:3d} | time: {time.time()-startt/60:5.2f}s | '
                    f'train loss {loss:5.2f} | valid loss {valid_loss:5.2f} ')
                print('-' * 89)

        else:
            print(f'Starting training with batch size:{self.opts.batch_size}')
            for epoch in range(self.opts.num_epochs):
                self.times = []
                startt= time.time()
                self.model.train()
                loss = self.train_one_epoch(epoch,save_every)
                self.model.eval()
                valid_loss = self.evaluate()
                print('-' * 89)
                print(f'| end of epoch {epoch:3d} | time: {time.time()-startt/60:5.2f}s | '
                    f'train loss {loss/len(self.train_dataset):5.2f} | valid loss {valid_loss:5.2f} ')
                print('-' * 89)

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
                                loss/ len(batch),(time.time()-running_time)/60)
            if epoch*self.opts.batch_size+i % save_every == save_every:
                self.save(epoch*self.opts.batch_size+i)
        self.scheduler.step()
        return running_loss


    def report_iteration(self,epoch,iter,lr,loss,time):
        self.times.append(time)
        print(f'| epoch {epoch:3d} | {iter:3d}/{int(self.num_batches)+1:5d} batches | '
                  f'lr {lr:02.2f} | {time*1000:5.2f} ms/batch  | avg loss {loss:5.2f} | '
                  f'ETA: {self.avg_time*(self.num_batches-iter):5.2f} min')

    def evaluate(self):
        loss = 0
        for i, batch in tqdm(enumerate(self.valid_iterator.batch_sampler),total=self.num_batches,desc='Loading batches...'):
            with torch.no_grad():
                src, tar = self.valid_dataset[batch]
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