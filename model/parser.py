#%%
from argparse import ArgumentParser
import torch

class HyperParameters(ArgumentParser):
    ENT_SIZE = 7
    base = 'datasets/idl/'
    types = ['train','valid','test']
    train_tgt = base + 'tgt-train.txt'
    train_src = base + 'src-train.txt'
    test_tgt = base + 'tgt-test.txt'
    test_src = base + 'src-test.txt'
    valid_tgt = base + 'tgt-valid.txt'
    valid_src = base + 'src-valid.txt'
    src_word_vec_size = 300
    tgt_word_vec_size = 300
    feat_vec_size = 300
    feat_vec_exponent = 1
    rnn_size = 600
    feat_merge = 'concat'
    dropout = [0.5]
    batch_size = 8
    num_epochs=10
    learning_rate = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "model/models/"

    def __init__(self) -> None:
        super().__init__(self)

# %%