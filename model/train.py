from torch import nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.dataset import IDLDataset
import torch
from model.parser import Opts
from model.model import DataToTextModel
import time


class Trainer():
    def __init__(self,opts: Opts, train_dataset: IDLDataset, train_iterator: DataLoader,
                valid_dataset: IDLDataset, valid_iterator: DataLoader,optim: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler, criterion: nn.Module, model: DataToTextModel = None, saved_model_path=None) -> None:
        self.opts = opts
        self.train_dataset = train_dataset
        self.train_iterator = train_iterator
        self.len_train = len(train_dataset)
        self.valid_dataset = valid_dataset
        self.valid_iterator = valid_iterator
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.criterion = criterion

    def train(self,save_every=1000):
        print(f'Starting training with batch size:{self.opts.batch_size}')
        for epoch in range(self.opts.num_epochs):
            startt= time.time()
            loss = self.train_one_epoch(epoch,save_every)
            valid_loss = self.evaluate()
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {time.time()-startt/60:5.2f}s | '
                f'train loss {loss:5.2f} | valid loss {valid_loss:5.2f} ')
            print('-' * 89)

    def train_one_epoch(self, epoch, save_every):
        running_loss = 0
        for i,batch in enumerate(self.train_iterator.batch_sampler):
            running_time = time.time()
            self.model.train()
            src, tar = self.train_dataset[batch]
            self.optim.zero_grad()
            out = self.model(src,tar)
            tar = tar.transpose(1,0)
            loss = 0
            for bn in range(self.opts.batch_size):
                loss += self.criterion(out[bn],tar[bn])
            loss.backward()
            self.optim.step()
            running_loss += loss.cpu().detach()
            self.report_iteration(epoch,i,self.scheduler.get_last_lr()[0],loss,(time.time()-running_time)/60)
            if epoch*self.opts.batch_size+i % save_every == 0:
                self.save(epoch,i)
        self.scheduler.step()
        return running_loss


    def report_iteration(self,epoch,iter,lr,loss,time):
        print(f'| epoch {epoch:3d} | {iter:5d}/{self.len_train:5d} batches | '
                  f'lr {lr:02.2f} | s/batch {time:5.2f} | loss {loss:5.2f} |'
                  f'ETA: {time*(self.len_train-iter)/self.opts.batch_size}')

    def evaluate(self):
        self.model.eval()
        loss = 0
        for i, batch in tqdm(enumerate(self.valid_iterator.batch_sampler),total=len(self.valid_dataset),desc='Loading batches...'):
            src, tar = self.valid_dataset[batch]
            out = self.model(src,tar)
            tar = tar.transp
            for bn in range(self.opts.batch_size):
                loss += self.criterion(out[bn],tar[bn])
        return loss

    def save(self,num):
        torch.save(self.model.state_dict(),self.opts.save_path+f'model_{num}.pt')