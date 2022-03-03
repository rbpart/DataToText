#%%
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

    dataset = IDLDataset(hparameters, 'train')
    test_dataset = IDLDataset(hparameters, 'test')

    model = build_model(hparameters,dataset)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tgt_vocab([dataset.tgt_pad_word])[0],
                        reduction="sum")
    optim = torch.optim.Adam(model.parameters(), lr=hparameters.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim,[lambda epoch: 0.95**(epoch/4)])

    # writer.add_graph(model)
    trainer = Trainer(hparameters, dataset, test_dataset,
                    optim, lr_scheduler, criterion, model, writer=writer)

    trainer.train()

# %%
