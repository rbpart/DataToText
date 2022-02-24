#%%
import torch.nn as nn
from model.parser import Opts
# from onmt.utils.loss import NMTLossCompute
# from onmt.utils.optimizers import Optimizer
# from onmt.utils.report_manager import ReportMgr
# from onmt.trainer import Trainer
# from onmt.utils.parse import ArgumentParser
# from onmt.utils.logging import init_logger
# from onmt.opts import train_opts
from model.build_model import build_model
from model.dataset import IDLDataset
from torch.utils.data import DataLoader
import torch
from model.train import Trainer

if __name__=="__main__":
    #logger = init_logger()
    opts = Opts()
    train_dataset = IDLDataset(opts, 'train')
    valid_dataset = IDLDataset(opts, 'valid')

    train_iterator = DataLoader(train_dataset, opts.batch_size, shuffle = True)
    valid_iterator = DataLoader(valid_dataset, opts.batch_size)

    model = build_model(opts,train_dataset)

    criterion = nn.NLLLoss(ignore_index=train_dataset.tgt_vocab([train_dataset.tgt_pad_word])[0],
                        reduction="sum")

    optim = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim,0.95)

    trainer = Trainer(opts, train_dataset, train_dataset, valid_dataset,
                    valid_iterator, optim, lr_scheduler, criterion, model)


# %%
