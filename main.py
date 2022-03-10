#%%
import torch.nn as nn
import torch
from model.parser import HyperParameters
from model.build_model import build_model
from model.dataset import IDLDataset
from model.trainer import Trainer
from torch.utils.tensorboard.writer import SummaryWriter
# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

if __name__=="__main__":
    writer = SummaryWriter()
    hparameters = HyperParameters()

    dataset = IDLDataset(hparameters, 'train')
    test_dataset = IDLDataset(hparameters, 'test')

    model = build_model(hparameters,dataset)
    # writer.add_graph(model,(src,tar,src_len,src_map))

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tgt_vocab([dataset.tgt_pad_word])[0],
                        reduction=hparameters.reduction)
    optim = torch.optim.Adam(model.parameters(), lr=hparameters.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim,[lambda epoch: 0.95**(epoch/100)])

    trainer = Trainer(hparameters, dataset, test_dataset,
                    optim, lr_scheduler, criterion, model, writer=writer)

    trainer.train(save_every=750, accumulate=4, clip=1.0)
