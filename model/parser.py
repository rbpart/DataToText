#%%
from argparse import ArgumentParser
import torch

class HyperParameters(ArgumentParser):
    ENT_SIZE = 10
    base = 'datasets/idl/'
    src_word_vocab = base + 'src_word.vocab.pt'
    src_feat_vocab = base + 'src_feat.vocab.pt'
    tgt_vocab = base + 'tgt_word.vocab.pt'
    types = ['train','valid','test']
    train_tgt = base + 'tgt-train.txt'
    train_src = base + 'src-train.txt'
    test_tgt = base + 'tgt-test.txt'
    test_src = base + 'src-test.txt'
    src_word_vec_size = 300
    tgt_word_vec_size = 300
    feat_vec_size = 300
    feat_vec_exponent = 1
    feat_merge = 'mlp'
    key_and_value = True
    dropout = [0.5]
    batch_size = 8
    num_epochs = 50
    learning_rate = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "model/models/"
    pretrained_tgt_embeddings_path = 'pretrained_embeddings/glove/glove.6B.300d.txt'
    train_size = 0.8
    reduction = "mean"
    accumulate=8
    clip =1.0
    use_pos = True
    forcing_frequency = 1
    tokenizer ='datasets/idl/tokenizer.pickle'

    def __init__(self) -> None:
        super().__init__(self)
        self.rnn_size = self.feat_vec_size+self.src_word_vec_size \
                                if self.feat_merge == 'concat' else 300

# %%