from collections import namedtuple
from typing import List, Tuple
import numpy as np
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
import torch
from tqdm import tqdm
from model.parser import HyperParameters
from torch.utils.data.sampler import Sampler

Entity =  namedtuple('Entity',('data','tags'))

Batch = namedtuple('Batch', ('src_vectors','src_words','tgt_vectors','tgt_words','tgt_lengths'))
class IDLDataset(Dataset):

    def __init__(self,opts:HyperParameters, type='train', transform = None) -> None:
        self.transform = transform
        self.opts = opts
        self.device = opts.device
        self.src_pad_word = '<blank>'
        self.src_pad_feat = '<blank>'
        self.tgt_pad_word = '<blank>'
        self.bos_word = '<bos>'
        self.eos_word = '<eos>'
        self.load_src(getattr(opts,f'{type}_src'))
        self.load_tgt(getattr(opts,f'{type}_tgt'))

    def __len__(self):
        return len(self.src_samples)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) is int:
            idx = [idx]
        src_raw = [self.src_samples[i] for i in idx]
        tar_tokens = [self.tgt_samples[i] for i in idx]
        src, tars = self._src_vector(src_raw),self._tgt_vector(tar_tokens)
        tar, tar_lengths, tokens = tars
        if self.transform:
            src = self.transform(src)
        return src.transpose(0,2), tar.transpose(0,1), tar_lengths, tokens, src_raw

    def _pad_entity(self,ent_word: list,ent_tag: list, size):
        blank_line_word = self.src_vocab([self.src_pad_word])*(size - len(ent_word))
        blank_line_feat = self.src_vocab_feat([self.src_pad_feat])*(size - len(ent_tag))
        return ent_word + blank_line_word, ent_tag + blank_line_feat

    def _pad_comment(self,list_comment: list):
        padding_size = (self.max_comment_len-len(list_comment))
        return list_comment + [self.tgt_pad_word]*padding_size, len(list_comment)

    def _src_vector(self,sample: List[List[Entity]]):
        words, tags = [], []
        if not isinstance(sample[0],list):
            sample = [sample]
        for record in sample:
            word, tag = [], []
            for entity in record:
                word_e, tag_e = self._pad_entity(self.src_vocab(entity.data),
                    self.src_vocab_feat(entity.tags), self.opts.ENT_SIZE)
                word += word_e
                tag += tag_e
            word, tag = self._pad_entity(word,tag,self.max_entities*self.opts.ENT_SIZE)
            words.append(word)
            tags.append(tag)
        src = torch.tensor([words,tags], dtype=torch.long, device=self.device) # [indexOfSample,indexOfEntity,indexOfFeature] = #indexWordInWordVocab => nb de mots
        return src

    def _tgt_vector(self,sample:List[str]):
        if not isinstance(sample[0],list):
            sample = [sample]
        paddeds, lenghts, tokens = [], [], []
        for comment in sample:
            padded_tokens, length_padding = self._pad_comment(comment)
            padded = self.tgt_vocab(padded_tokens)
            paddeds.append(padded)
            lenghts.append(length_padding)
            tokens.append(padded_tokens)
        lenghts = torch.tensor([[s] for s in lenghts], dtype=torch.long, device=self.device)
        return  torch.tensor(paddeds, dtype=torch.long, device=self.device), lenghts, tokens

    def _load_entity(self, entity:str, info_token = ' ', split_token = '|') -> Tuple:
        data, tags = [], []
        for info in entity.split(info_token):
            data.append(info.split(split_token)[0])
            tags.append(info.split(split_token)[1])
        return data, tags

    def _load_entities(self,line:str,entity_token = '<ent>|<ent>') -> List[Entity]:
        return [Entity(*self._load_entity(entity.strip())) for entity in line.split(entity_token) if entity != '']

    def load_src(self,path):
        with open(path,'r') as file:
            samples = [self._load_entities(line.strip()) for line in tqdm(file.readlines()) if line != '']
        self.max_entities = np.max([len(samp) for samp in samples])
        self.src_samples : List[Entity] = samples
        self.src_vocab : Vocab = torch.load(self.opts.src_word_vocab)
        self.src_vocab_feat : Vocab = torch.load(self.opts.src_feat_vocab)

    def _process(self,line:str):
        return line.split(' ')

    def load_tgt(self,path):
        with open(path,'r') as file:
            samples = [self._process(line) for line in tqdm(file.readlines()) if line != '']
        self.max_comment_len = np.max([len(samp) for samp in samples])
        self.tgt_samples : List[str] = samples
        self.tgt_vocab : Vocab = torch.load(self.opts.tgt_vocab)
        self.len_tgt_vocab = len(self.tgt_vocab)

class BatchSamplerSimilarLength(Sampler):
    def __init__(self, dataset, batch_size, indices=None, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # get the indicies and length
        self.indices = [(i, tgt_len) for i, (src, tgt, tgt_len, tgt_tokens) in enumerate(dataset)]
        # if indices are passed, then use only the ones passed (for ddp)
        if indices is not None:
            self.indices = torch.tensor(self.indices)[indices].tolist()

    def __iter__(self):
        if self.shuffle:
            torch.random.shuffle(self.indices)

        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(self.indices), self.batch_size * 100):
            pooled_indices.extend(sorted(self.indices[i:i + self.batch_size * 100], key=lambda x: x[1]))
        self.pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        batches = [self.pooled_indices[i:i + self.batch_size] for i in
                   range(0, len(self.pooled_indices), self.batch_size)]

        if self.shuffle:
            torch.random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.pooled_indices) // self.batch_siz