from torch import nn as nn
import torch
from decoders.hierarchical_decoder import HierarchicalRNNDecoder
from encoders.hierarchical_transformer import HierarchicalTransformerEncoder
from tqdm import tqdm
from model.utils import bleu_score_
from model.dataset import IDLDataset

class DataToTextModel(nn.Module):
    def __init__(self, encoder: HierarchicalTransformerEncoder, decoder: HierarchicalRNNDecoder, generator: nn.Sequential,
        dataset_model: IDLDataset = None, device = None) -> None:
        super().__init__()
        self.base_src_word_vocab = None
        self.src_feat_vocab = None
        self.tgt_word_vocab = None
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

    def forward(self, src, tgt, lengths = None, src_map = None, bptt=False, with_align=False):
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
        enc_state, memory_bank, lengths = self.encoder(src, lengths = lengths)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                    memory_lengths=lengths,
                                    with_align=with_align)
        if "std" in attns: # et la copy attention elle sert a quoi ?
            attn = attns["std"]
            attn_key = 'std'

        return  self.generator(dec_out,attn,src_map).transpose(0,1)


    #Utilitaries for the model

    def _select_dataset(self,dataset):
        if dataset is not None:
            dataset = dataset
        elif self.dataset is not None:
            dataset = self.dataset
        else:
            raise RuntimeError('Provide the dataset with the vocab used for training or initialize model with the dataset')
        return dataset

    def infer(self, src, src_len, src_map,vocab,dataset = None):
        dataset = self._select_dataset(dataset)
        _,batch_size,_ = src.shape
        beggining = [vocab(t) for t in [['<bos>']*batch_size]]
        tgt = torch.zeros((dataset.max_comment_len,batch_size), dtype=torch.long, device=self.device, requires_grad = False)
        tgt[0,:] = torch.tensor(beggining, device=self.device, requires_grad = False)
        self.eval()
        with torch.no_grad():
            for i in tqdm(range(1,100)):
                out = self.forward(src,
                            torch.where(tgt[:i,:] < len(dataset.tgt_vocab),tgt[:i,:],0),
                            src_len,
                            src_map=src_map)
                tgt[i,:] = out.argmax(2).transpose(0,1)[i-1,:] #maybe problem with this line cause repetition
        self.train()
        return tgt

    def infer_to_sentence(self, src, src_len, src_map, vocab, dataset=None):
        dataset = self._select_dataset(dataset)
        tgts = self.infer(src, src_len, src_map, vocab, dataset)
        tgts = tgts.transpose(0,1).tolist()
        comments = []
        for sentence in tgts:
            comments.append(' '.join(vocab.lookup_tokens(sentence)))
        return comments

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    @staticmethod
    def metrics(sentences, golds):
        BLEU = bleu_score_(sentences,golds)
        return BLEU