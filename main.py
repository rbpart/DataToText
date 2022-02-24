#%%
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from model.parser import HyperParameters
from model.build_model import build_model
from model.dataset import IDLDataset
from model.trainer import Trainer
from torch.utils.tensorboard.writer import SummaryWriter

if __name__=="__main__":
    writer = SummaryWriter()
    hparameters = HyperParameters()

    train_dataset = IDLDataset(hparameters, 'train')
    valid_dataset = IDLDataset(hparameters, 'valid')
    test_dataset = IDLDataset(hparameters, 'test')

    model = build_model(hparameters,train_dataset)
    criterion = nn.NLLLoss(ignore_index=train_dataset.tgt_vocab([train_dataset.tgt_pad_word])[0],
                        reduction="sum")
    optim = torch.optim.Adam(model.parameters(), lr=hparameters.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim,lambda epoch: 0.99**epoch)


    trainer = Trainer(hparameters, train_dataset, valid_dataset, test_dataset,
                    optim, lr_scheduler, criterion, model)

    trainer.train()
# %%
