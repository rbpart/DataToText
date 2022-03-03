#%%
import torch.nn as nn
import torch
from model.parser import HyperParameters
from model.build_model import build_model
from model.dataset import IDLDataset
from model.trainer import Trainer
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
if __name__=="__main__":
    writer = SummaryWriter()
    hparameters = HyperParameters()

    dataset = IDLDataset(hparameters, 'train')
    test_dataset = IDLDataset(hparameters, 'test')

    model = build_model(hparameters,dataset)
    # model.load_params('model/models/checkpoint_shitty_model_42epoch/model_shitty_model_42epoch.pt')
    src,tar,leng,toks,src_raws = dataset[[0,1]]
    outputs,vocab = model.infer_to_sentence(src,src_raws,dataset)
