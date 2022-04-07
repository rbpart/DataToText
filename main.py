#%%
import torch.nn as nn
import torch
from model.parser import HyperParameters
from model.build_model import build_model
from model.dataset import IDLDataset, BatchSamplerSimilarLength
from model.trainer import Trainer
from torch.utils.tensorboard.writer import SummaryWriter

if __name__=="__main__":
    writer = SummaryWriter()
    hparameters = HyperParameters()
    dataset = IDLDataset(hparameters, 'train')
    test_dataset = IDLDataset(hparameters, 'test')
    #sampler = BatchSamplerSimilarLength(dataset,hparameters.batch_size,shuffle=True)
    model = build_model(hparameters,dataset)
    # writer.add_graph(model,dataset[0:2])

    criterion = nn.CrossEntropyLoss(
                        ignore_index=dataset.tgt_vocab([dataset.tgt_pad_word])[0],
                        reduction=hparameters.reduction)
    optim = torch.optim.Adam(model.parameters(), lr=hparameters.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,patience=4,threshold=2e-3,factor=0.5)

    trainer = Trainer(hparameters, dataset, test_dataset,
                    optim, lr_scheduler, criterion,
                    model, writer=writer, create_experiment=True)

    trainer.train(save_every=1500,
                accumulate=hparameters.accumulate,
                clip=hparameters.clip,
                validate_every=1,
                augment_size=0)
# %%
