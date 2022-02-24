#%%
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from model.parser import Opts
from model.build_model import build_model
from model.dataset import IDLDataset
from model.train import Trainer
from torch.utils.tensorboard.writer import SummaryWriter

if __name__=="__main__":
    writer = SummaryWriter()
    opts = Opts()
    train_dataset = IDLDataset(opts, 'train')
    valid_dataset = IDLDataset(opts, 'valid')

    train_iterator = DataLoader(train_dataset, opts.batch_size, shuffle = True)
    valid_iterator = DataLoader(valid_dataset, opts.batch_size)

    model = build_model(opts,train_dataset)

    criterion = nn.NLLLoss(ignore_index=train_dataset.tgt_vocab([train_dataset.tgt_pad_word])[0],
                        reduction="sum")

    optim = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim,lambda epoch: 0.99**epoch)

    trainer = Trainer(opts, train_dataset, train_iterator, valid_dataset,
                    valid_iterator, optim, lr_scheduler, criterion, model)

    trainer.train()
# %%
