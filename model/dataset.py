from collections import namedtuple
import pickle
from typing import List, Tuple
import numpy as np
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
import torch
import re
from tqdm import tqdm
from model.parser import HyperParameters
from torch.utils.data.sampler import Sampler
from torch.nn.functional import one_hot
from torchtext.vocab.vocab_factory import build_vocab_from_iterator
from torchtext.vocab.vocab import Vocab
from itertools import chain
from copy import copy, deepcopy
import nlpaug.augmenter.word as naw
from nltk.tokenize import word_tokenize

Entity =  namedtuple('Entity',('data','tags'))

class Source():
    def __init__(self, tensor: torch.Tensor, lens: torch.Tensor,
                map: torch.Tensor, raw: List[List[Entity]]) -> None:
        self.tensor = tensor
        self.lens = lens
        self.map = map
        self.raw = raw
class Target():
    def __init__(self, tensor: torch.Tensor, lens: torch.Tensor,
                raw: List[List[str]]) -> None:
        self.tensor = tensor
        self.lens = lens
        self.raw = raw
class Batch():
    def __init__(self, source: Source, vocab: Vocab, target: Target = None) -> None:
        self.source = source
        self.target = target
        self.vocab = vocab

class IDLDataset(Dataset):

    def __init__(self,opts:HyperParameters, type='train', transform = None) -> None:
        self.transform = transform
        self.opts = opts
        self.device = opts.device
        self.tokenizer = pickle.load(open(opts.tokenizer,'rb'))
        self.src_pad_word = '<blank>'
        self.src_pad_feat = '<blank>'
        self.tgt_pad_word = '<blank>'
        self.bos_word = '<bos>'
        self.eos_word = '<eos>'
        self.load_src(getattr(opts,f'{type}_src'))
        self.load_tgt(getattr(opts,f'{type}_tgt'))
        self.bos_word_idx = self.tgt_vocab[self.bos_word]
        self.eos_word_idx = self.tgt_vocab[self.eos_word]
        self.pad_word_idx = self.tgt_vocab[self.tgt_pad_word]


    def __len__(self):
        return len(self.src_samples)

    def __getitem__(self,idx) -> Batch:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) is int:
            idx = [idx]
        if type(idx) is slice:
            step = idx.step if idx.step is not None else 1
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else self.__len__()
            idx = [i for i in range(start,stop,step)]
        src_raw = [self.src_samples[i] for i in idx]
        tar_tokens = [self.tgt_samples[i] for i in idx]
        self.max_comment_len = np.max([len(samp) for samp in tar_tokens])
        self.max_entities = np.max([len(sr) for sr in src_raw])
        srcs, tars = self._src_vector(src_raw),self._tgt_vector(tar_tokens)
        tar, tar_lengths, tokens = tars
        src, src_len = srcs
        src_map, fusedvocab = self._gen_src_map(src_raw)
        if self.transform:
            src = self.transform(src)
        src = src.transpose(0,2)
        tar = tar.transpose(0,1)
        source = Source(src,src_len,src_map,src_raw)
        target = Target(tar,tar_lengths,tokens)
        return Batch(source, fusedvocab, target)

    def _pad_entity(self,ent_word: list,ent_tag: list, size):
        blank_line_word = self.src_vocab([self.src_pad_word])*(size - len(ent_word))
        blank_line_feat = self.src_vocab_feat([self.src_pad_feat])*(size - len(ent_tag))
        return ent_word + blank_line_word, ent_tag + blank_line_feat, len(ent_word)

    def _pad_comment(self,list_comment: list):
        padding_size = (self.max_comment_len-len(list_comment))
        return list_comment + [self.tgt_pad_word]*padding_size, len(list_comment)

    def _src_vector(self,sample: List[List[Entity]]):
        words, tags, lenghts = [], [], []
        if not isinstance(sample[0],list):
            sample = [sample]
        for record in sample:
            word, tag = [], []
            for entity in record:
                word_e, tag_e,_ = self._pad_entity(self.src_vocab(entity.data),
                    self.src_vocab_feat(entity.tags), self.opts.ENT_SIZE)
                word += word_e
                tag += tag_e
            word, tag, leng = self._pad_entity(word,tag,self.max_entities*self.opts.ENT_SIZE)
            lenghts.append(leng)
            words.append(word)
            tags.append(tag)
        src = torch.tensor([words,tags], dtype=torch.long, device=self.device)
        lenghts = torch.tensor(lenghts,device = self.device, dtype=torch.long)
        return src, lenghts

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

    def _preprocess_src(self,data):
        return [string.lower().strip().strip('_') for string in data]

    def _load_entity(self, entity:str, info_token = ' ', split_token = '|') -> Tuple:
        data, tags = ['<ent>'], ['<ent>']
        for info in entity.split(info_token):
            data.append(info.split(split_token)[0])
            tags.append(info.split(split_token)[1])
        return self._preprocess_src(data), tags

    def _load_entities(self,line:str,entity_token = '<ent>|<ent>') -> List[Entity]:
        return [Entity(*self._load_entity(entity.strip())) for entity in line.split(entity_token) if entity != '']

    def load_src(self,path):
        with open(path,'r') as file:
            samples = [self._load_entities(line.strip()) for line in tqdm(file.readlines()) if line != '']
        self.max_entities = np.max([len(samp) for samp in samples])
        self.src_samples : List[Entity] = samples
        self.base_src = deepcopy(self.src_samples)
        self.src_vocab : Vocab = torch.load(self.opts.src_word_vocab)
        self.src_vocab_feat : Vocab = torch.load(self.opts.src_feat_vocab)

    def _process(self,line:str):
        s = line.replace(r'\n','').lower()
        s = re.sub('<eos>','',s)
        s = re.sub('<bos>','',s)
        s = re.sub('([.,!?()])', r'\1 ', s)
        s = re.sub('\s{2,}', ' ', s)
        tokenized = self.tokenizer.tokenize(word_tokenize(s))
        return ['<bos>'] + tokenized + ['<eos>']

    def load_tgt(self,path):
        with open(path,'r') as file:
            samples = [self._process(line) for line in tqdm(file.readlines()) if line != '']
        self.max_comment_len = np.max([len(samp) for samp in samples])
        self.tgt_samples : List[str] = samples
        self.base_tgt = deepcopy(self.tgt_samples)
        self.tgt_vocab : Vocab = torch.load(self.opts.tgt_vocab)

    def _gen_src_map(self,src_words):
        data = [list(chain.from_iterable([ent.data for ent in sample])) for sample in src_words]
        data_vocab = build_vocab_from_iterator(data)
        fused_vocab = self.fuse_vocabs(self.tgt_vocab,data_vocab)
        max_len = self.max_entities*self.opts.ENT_SIZE
        data = [fused_vocab(sample) + fused_vocab(['<blank>'])*(max_len-len(sample)) for sample in data]
        with torch.autocast(device_type=self.device):
            src_map = torch.tensor(data, dtype=torch.long, device=self.device)
            src_map = one_hot(src_map).transpose(0,1).to(torch.float).to_sparse()
        return src_map, fused_vocab

    def fuse_vocabs(self,vocab_base: Vocab,vocab_extension: Vocab):
        newvocab = deepcopy(vocab_base)
        itos = vocab_base.get_itos()
        itos_ex = vocab_extension.get_itos()
        for w in (set(itos_ex) - set(itos)):
            newvocab.append_token(w)
        return newvocab

    def augment_data(self, indices, size = 0.7):
        #aug = naw.BackTranslationAug(max_length=self.max_comment_len) #inutilisable sans plus gros gpu
        aug = naw.SynonymAug(aug_src='wordnet',aug_max=10)
        self.src_samples, self.tgt_samples = copy(self.base_src), copy(self.base_tgt)
        picks = np.random.choice(indices,size=int(size*len(indices)),replace=True).astype(int).tolist()
        n = len(self.src_samples)
        processed = []
        for i,pick in tqdm(enumerate(picks)):
            translated = aug.augment(' '.join(self.tgt_samples[pick]))
            processed += self._process(translated)
            self.src_samples.append(self.src_samples[pick])
            self.tgt_samples.append(self._process(translated))
            indices.append(n+i)
        print(f'final train size: {len(self.src_samples)}')
        self.tgt_vocab = self.fuse_vocabs(self.tgt_vocab, build_vocab_from_iterator(processed))
        self.len_tgt_vocab = len(self.tgt_vocab)
        return indices

class BatchSamplerSimilarLength(Sampler):
    def __init__(self, dataset: IDLDataset, batch_size, indices=None, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # get the indices and length
        self.indices = [(i, batch.source.lens) for i, batch in enumerate(dataset)]
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
        return len(self.pooled_indices) // self.batch_size