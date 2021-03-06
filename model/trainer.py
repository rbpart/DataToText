from functools import reduce
from torch import nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset
from model.dataset import IDLDataset
import torch
from model.parser import HyperParameters
from model.model import DataToTextModel
from model.build_model import build_decoder
import time
import os
import shutil
from torch.utils.data import random_split
from torch.utils.data.sampler import Sampler
from model.utils import bleu_score_


class Trainer():
    def __init__(self, opts: HyperParameters = None, dataset: IDLDataset = None,
                train_dataset: IDLDataset = None,valid_dataset: IDLDataset = None, test_dataset: IDLDataset = None,
                optim: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                criterion: nn.Module = None, model: DataToTextModel = None, checkpoint_file = None,
                writer: SummaryWriter = None, sampler: Sampler = None, create_experiment = False) -> None:

        self.checkpoint_file = checkpoint_file
        self.writer = writer
        if checkpoint_file:
            self.load(checkpoint_file)
        elif None not in [model,optim,scheduler]:
            self.model = model
            self.optim = optim
            self.scheduler = scheduler
        else:
            raise ValueError('Either provide a path to load checkpoint or provide all the required arguments to create a new model')

        if None not in [opts, criterion]:
            self.opts = opts
            self.criterion = criterion
            self.dataset =dataset
            self.train_dataset = train_dataset
            self.valid_dataset = valid_dataset
        else:
            raise ValueError('Criterion and train and valid datasets should always be provided')

        self.sampler = sampler
        self.test_dataset = test_dataset
        if test_dataset:
            self.test_iterator = DataLoader(test_dataset, opts.batch_size)
        if create_experiment:
            self.create_experiment()
        self.build_iterators()

    @property
    def avg_time(self):
        return np.mean(self.times)

    def create_experiment(self):
        experiments = os.listdir(self.opts.save_path)
        experiments = [int(ex.split('t')[-1]) for ex in experiments]
        maxi = max(experiments)
        newexp = f'experiment{maxi+1}'
        os.mkdir(self.opts.save_path+newexp)
        self.opts.save_path += newexp
        os.mkdir(self.opts.save_path + '/' + self.writer.log_dir.split('/')[-1])
        self.runfile = self.writer.log_dir + '/' + os.listdir(self.writer.log_dir)[0]
        shutil.copyfile('model/parser.py','model/models/'+newexp+'/parser.py')

    def train_k_fold(self, nfolds, save_every = 1000, clip = None, accumulate = 1):
        folds = self.train_valid_split(folds=nfolds)
        for i in range(folds):
            self.train_dataset = ConcatDataset(folds[0:i] + folds[i+1,nfolds])
            self.train_iterator = DataLoader(self.train_dataset, self.opts.batch_size,
                                    shuffle=True if not self.sampler else False,
                                    sampler = self.sampler, drop_last= True,
                                    pin_memory= True if self.opts.device == 'cuda' else False)
            self.valid_dataset = folds[i]
            self.valid_iterator = DataLoader(self.valid_dataset, self.opts.batch_size/2)
            self.num_batches = int(len(self.train_dataset)/self.opts.batch_size)
            self.train(save_every, clip = clip, accumulate = accumulate)
        return

    def train(self,save_every=1000, clip = None, accumulate = 1, validate_every=1):
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
            train_loss = self.train_one_epoch(epoch,save_every,accumulate = accumulate, clip = clip)
            if epoch % validate_every == 0:
                self.model.eval()
                valid_loss = self.evaluate(self.valid_dataset,self.valid_iterator)
                scores = self.test(epoch)
                self.report_epoch(epoch,start_time,train_loss,valid_loss,scores)

        self.save('final')
        if self.test_dataset:
            self.evaluate(self.test_dataset,self.test_iterator)

    def train_one_epoch(self, epoch, save_every, clip = None, accumulate = 1):
        running_loss = 0
        clipped = 1
        unclipped = 1
        self.optim.zero_grad()
        for i,batch in enumerate(self.train_iterator.batch_sampler,1):
            global_step = epoch*self.num_batches + i
            running_time = time.time()
            batch_ = self.train_dataset[batch]
            out,attns = self.model(batch_)
            batch_.target.tensor = batch_.target.tensor.transpose(1,0)
            loss = self.criterion(out.transpose(1,2),batch_.target.tensor[:,1:])
            if self.opts.reduction == "mean":
                loss /= accumulate
            loss.backward()
            loss_ = loss.detach().cpu()
            running_loss += loss_
            if i % accumulate == 0 or (i+1) == len(self.train_iterator.batch_sampler):
                if isinstance(clip,float):
                    unclipped = nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), clip)
                    clipped = np.sqrt(np.sum([p.grad.detach().cpu().data.norm(2)**2 for p in self.model.parameters()]))
                self.optim.step()
                self.optim.zero_grad()
                loss = 0
            self.report_iteration(epoch,i,self.optim.param_groups[0]['lr'],
                                loss_,(time.time()-running_time)/60, unclipped, clipped)
            if global_step % save_every == 0:
                self.save(global_step)
        self.scheduler_(running_loss/len(self.train_dataset))
        return running_loss * accumulate

    def report_iteration(self,epoch,iter,lr,loss,time, unclipped, clipped):
        self.times.append(time)
        shutil.copy(self.runfile,self.opts.save_path + '/' + self.writer.log_dir.split('/')[-1])
        print(f'| epoch {epoch:3d} | {iter:3d}/{int(self.num_batches)+1:3d} batches | '+
                  f'lr {lr:1.5f} | {time*1000:5.2f} ms/batch  | avg loss {loss:5.2f} | '+
                  f'ETA: {self.avg_time*(self.num_batches-iter):5.2f} min', end='\r')
        if self.writer is not None:
            self.writer.add_scalar('Training/Loss',loss,iter+epoch*self.num_batches)
            self.writer.add_scalar('Training/UnclippedGrad',unclipped,iter+epoch*self.num_batches)
            self.writer.add_scalar('Training/NormOfClippedGrad',clipped,iter+epoch*self.num_batches)

    def report_epoch(self,epoch,start_time,train_loss,valid_loss,scores):
            print('-' * 100)
            print(f'| end of epoch {epoch:3d} | elapsed time: {(time.time()-start_time)/60:3.2f} min | '
                f'avg train loss {train_loss/len(self.train_dataset):5.2f} | avg valid loss {valid_loss/len(self.valid_dataset):5.2f} ')
            print('-' * 100)
            if self.writer is not None:
                self.writer.add_scalar('Eval/AvgLossTrain',train_loss/len(self.train_dataset),epoch)
                self.writer.add_scalar('Eval/AvgLossValid',valid_loss/len(self.valid_dataset),epoch)
                for k,v in scores.items():
                    self.writer.add_scalar(f'Eval/{k}Test',v,epoch)

    def evaluate(self, dataset, iterator: DataLoader):
        loss = 0
        with torch.no_grad():
            for batch in iterator.batch_sampler:
                batch_ = dataset[batch]
                out,attns = self.model(batch_)
                batch_.target.tensor = batch_.target.tensor.transpose(1,0)
                loss += self.criterion(out.transpose(1,2),batch_.target.tensor[:,1:])
        return loss

    def test(self, epoch, test_size=4):
        with torch.no_grad():
            scores = {}
            batch = torch.randint(0,len(self.test_dataset),(test_size,))
            batch_ = self.test_dataset[batch]
            batch_.target.tensor = batch_.target.tensor.transpose(1,0)
            outputs = self.model.infer_to_sentence(batch_,
                                                self.dataset.bos_word_idx,
                                                self.dataset.eos_word_idx,
                                                predictions=batch_.target.lens.max(),
                                                beam_width=10, best = True,
                                                block_n=3, progress_bar=False)
            scores['bleu'] = bleu_score_(outputs,batch_.target.raw)
            targets = '  \n'.join([' '.join(s) for s in batch_.target.raw]).replace('<blank>','')
            self.writer.add_text('Eval/Outputs','  \n'.join(outputs),epoch)
            self.writer.add_text('Eval/Target',targets,epoch)
            return scores

    def build_iterators(self):
        self.train_iterator = DataLoader(self.train_dataset,
        self.opts.batch_size,shuffle=True,
        pin_memory=True)
        self.valid_iterator = DataLoader(self.valid_dataset,self.opts.batch_size)
        self.num_batches = int(len(self.train_dataset)/self.opts.batch_size)

    def scheduler_(self, value = None):
        if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
            self.scheduler.step(value)
        elif self.scheduler is not None:
            self.scheduler.step()

    def save(self,num):
        checkpoint_file = self.opts.save_path+'/'+f'checkpoint_{num}/'
        os.mkdir(checkpoint_file)
        torch.save(self.model.state_dict(),checkpoint_file+f'model_{num}.pt')
        torch.save(self.optim.state_dict(),checkpoint_file+f'optim_{num}.pt')
        torch.save(self.scheduler.state_dict(),checkpoint_file+f'scheduler_{num}.pt')

    def load(self):
        num = self.checkpoint_file.split('_')[0]
        self.optim = torch.load('model/models/'+self.checkpoint_file+f'/optim_{num}.pt')
        self.model = torch.load('model/models/'+self.checkpoint_file+f'/model_{num}.pt')
        self.scheduler = torch.load('model/models/'+self.checkpoint_file+f'/scheduler_{num}.pt')