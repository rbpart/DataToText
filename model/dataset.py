from collections import namedtuple
from typing import List, Tuple, Union
import numpy as np
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator
import torch
from tqdm import tqdm
from model.parser import Opts

Entity =  namedtuple('Entity',('data','tags'))

class IDLDataset(Dataset):

    def __init__(self,opts:Opts, type='valid', transform = None) -> None:
        self.transform = transform
        self.opts = opts
        self.device = opts.device
        self.src_pad_word = '<blank>'
        self.src_pad_feat = '<blank>'
        self.tgt_pad_word = '<blank>'
        self.load_src(getattr(opts,f'{type}_src'))
        self.load_tgt(getattr(opts,f'{type}_tgt'))

    def __len__(self):
        return len(self.src_samples)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) is int:
            idx = [idx]
        src, tar = [self.src_samples[i] for i in idx], [self.tgt_samples[i] for i in idx]
        src, tar = self._src_vector(src),self._tgt_vector(tar)
        if self.transform:
            src = self.transform(src)
        return src.transpose(0,2), tar.transpose(0,1)

    def _pad(self,ent_word: list,ent_tag: list, size):
        blank_line_word = self.src_vocab([self.src_pad_word])*(size - len(ent_word))
        blank_line_feat = self.src_vocab_feat([self.src_pad_feat])*(size - len(ent_tag))
        return ent_word + blank_line_word, ent_tag + blank_line_feat

    def _pad_comment(self,list_comment: list):
        return list_comment + self.tgt_vocab([self.tgt_pad_word])*(self.max_comment_len-len(list_comment))

    def _src_vector(self,sample: Union[List[Entity],List[List[Entity]]]):
        words, tags = [], []
        if not isinstance(sample[0],list):
            sample = [sample]
        for record in sample:
            word, tag = [], []
            for entity in record:
                word_e, tag_e = self._pad(self.src_vocab(entity.data),
                self.src_vocab_feat(entity.tags), self.opts.ENT_SIZE)
                word += word_e
                tag += tag_e
            word, tag = self._pad(word,tag,self.max_entities*self.opts.ENT_SIZE)
            words.append(word)
            tags.append(tag)
        src = torch.LongTensor([words,tags]).to(self.device) # [indexOfSample,indexOfEntity,indexOfFeature] = #indexWordInWordVocab => nb de mots
        return src

    def _tgt_vector(self,sample:List[str]):
        if isinstance(sample[0],list):
            return  torch.LongTensor([self._pad_comment(self.tgt_vocab(comment)) for comment in sample]).to(self.device)
        return torch.LongTensor([self._pad_comment(self.tgt_vocab(sample))]).to(self.device)

    def _load_entity(self, entity:str, info_token = ' ', split_token = '|') -> Tuple:
        data, tags = [], []
        for info in entity.split(info_token):
            data.append(info.split(split_token)[0])
            tags.append(info.split(split_token)[1])
        return data, tags

    def _load_entities(self,line:str,entity_token = '<ent>|<ent>') -> List[Entity]:
        return [Entity(*self._load_entity(entity.strip())) for entity in line.split(entity_token) if entity != '']

    def _unique_tags(self,samples):
        unique_tags = []
        for sample in np.random.choice(samples,size = int(len(samples)/10)):
            for ent in sample:
                unique_tags += ent.tags
        return np.unique(unique_tags).tolist()

    def load_src(self,path):
        with open(path,'r') as file:
            samples = [self._load_entities(line.strip()) for line in tqdm(file.readlines()) if line != '']
        self.max_entities = np.max([len(samp) for samp in samples])
        self.src_samples : List[Entity] = samples
        self.src_vocab : Vocab = build_vocab_from_iterator([entity.data for sample in samples for entity in sample ])
        self.src_vocab_feat : Vocab = build_vocab_from_iterator([entity.tags for sample in samples for entity in sample ])

    def _process(self,line:str):
        return line.split(' ')

    def load_tgt(self,path):
        with open(path,'r') as file:
            samples = [self._process(line) for line in tqdm(file.readlines()) if line != '']
        self.max_comment_len = np.max([len(samp) for samp in samples])
        self.tgt_samples : List[str] = samples
        self.tgt_vocab : Vocab = build_vocab_from_iterator(samples)
        self.tgt_vocab.insert_token(self.tgt_pad_word,len(self.tgt_vocab))
        self.len_tgt_vocab = len(self.tgt_vocab)
