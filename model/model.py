from torch import nn as nn
import torch
from decoders.hierarchical_decoder import HierarchicalRNNDecoder
from encoders.hierarchical_transformer import HierarchicalTransformerEncoder
from tqdm import tqdm
from torchtext.vocab.vocab_factory import build_vocab_from_iterator
from collections import Counter
from itertools import chain
import copy
from model.dataset import IDLDataset


class DataToTextModel(nn.Module):
    def __init__(self, encoder: HierarchicalTransformerEncoder, decoder: HierarchicalRNNDecoder, generator: nn.Sequential,
        dataset_model: IDLDataset = None, device = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.dataset = dataset_model
        if device is not None:
            self.to(device)
            self.device = device
        else:
            self.device = 'cpu'

    def load_params(self, path):
        tensors = torch.load(path)
        for k in tensors.keys():
            self._parameters[k] = tensors[k]

    def forward(self, src, tgt, lengths = None, bptt=False, with_align=False, src_words = None, inference = False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        enc_state, memory_bank, lengths = self.encoder(src, lengths= None)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                    memory_lengths=lengths,
                                    with_align=with_align)
        if "std" in attns:
            attn = attns["std"]
            attn_key = 'std'
        # Pour générer src_map, il va falloir créer un "oov vocab" du batch et faire
        # ext_vocab = Vocab(Counter(src))
        # src_map = torch.LongTensor([src_ex_vocab.stoi[w] for w in src])
        if src_words is not None:
            data = [list(chain.from_iterable([ent.data for ent in sample])) for sample in src_words]
            data_vocab = build_vocab_from_iterator(data)
            fused_vocab = self.fuse_vocabs(self.dataset.tgt_vocab,data_vocab)
            max_len =  attn.size()[2]
            data = [fused_vocab(sample) + fused_vocab(['<blank>'])*(max_len-len(sample)) for sample in data]
            src_map = torch.tensor(data, dtype=torch.long, device=self.device)
            src_map = nn.functional.one_hot(src_map).transpose(0,1).to(torch.float)

        if inference:
            return self.generator(dec_out,attn,src_map).transpose(0,1), fused_vocab
        return  self.generator(dec_out,attn,src_map).transpose(0,1)


    #Utilitaries for the model

    def fuse_vocabs(self,vocab_base,vocab_extension):
        newvocab = copy.copy(vocab_base)
        itos = vocab_base.get_itos()
        itos_ex = vocab_extension.get_itos()
        for w in (set(itos_ex) - set(itos)):
            newvocab.append_token(w)
        return newvocab

    def _select_dataset(self,dataset):
        if dataset is not None:
            dataset = dataset
        elif self.dataset is not None:
            dataset = self.dataset
        else:
            raise RuntimeError('Provide the dataset with the vocab used for training or initialize model with the dataset')
        return dataset

    def infer(self,src,src_words,dataset = None):
        dataset = self._select_dataset(dataset)
        _,batch_size,_ = src.shape
        beggining = [dataset.tgt_vocab(t) for t in [['<bos>']*batch_size]]
        tgt = torch.zeros((dataset.max_comment_len,batch_size), dtype=torch.long, device=self.device, requires_grad = False)
        tgt[0,:] = torch.tensor(beggining, device=self.device, requires_grad = False)
        self.eval()
        with torch.no_grad():
            for i in tqdm(range(1,100)):
                out,vocab = self.forward(src,torch.where(
                    tgt[:i,:] < len(dataset.tgt_vocab),
                    tgt[:i,:],
                    0), src_words=src_words, inference=True)
                tgt[i,:] = out.argmax(2).transpose(0,1)[i-1,:] #maybe problem with this line cause repetition
        self.train()
        return tgt, vocab

    def infer_to_sentence(self, src, src_words, dataset=None):
        dataset = self._select_dataset(dataset)
        tgts, vocab= self.infer(src,src_words,dataset)
        tgts = tgts.transpose(0,1).tolist()
        comments = []
        for sentence in tgts:
            comments.append(' '.join(vocab.lookup_tokens(sentence)))
        return comments

    def metrics(sentences, golds):
        return