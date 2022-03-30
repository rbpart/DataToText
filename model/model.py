from torch import nn as nn
import torch
from decoders.hierarchical_decoder import HierarchicalRNNDecoder
from encoders.hierarchical_transformer import HierarchicalTransformerEncoder
from tqdm import tqdm
from model.utils import bleu_score_
from model.dataset import IDLDataset, Batch
import model.output_to_text as ott
from torchtext.vocab import Vocab
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

    def forward(self, batch: Batch, with_align=False, hidden_state=None, inference = False) -> torch.Tensor:
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
        if not inference:
            dec_in = batch.target.tensor[:-1]
        else:
            dec_in = batch.target.tensor
        enc_state, memory_bank, lengths = self.encoder(batch.source.tensor , lengths = batch.source.lens)

        if hidden_state is not None:
            self.decoder.state['hidden'] = hidden_state
        else:
            self.decoder.init_state(batch.source.tensor, memory_bank, enc_state)

        dec_out, attns = self.decoder(dec_in, memory_bank,
                                    memory_lengths=lengths,
                                    with_align=with_align)
        if "std" in attns: # et la copy attention elle sert a quoi ?
            attn = attns["std"]
            attn_key = 'std'
        return  self.generator(dec_out,attn,batch.source.map).transpose(0,1)

    #Utilitaries for the model

    def infer(self, batch: Batch,
                bos_idx, eos_idx, inference_type: str = 'beam_search', **kwargs):
        if inference_type not in ['beam_search','greedy_search']:
            raise ValueError(f'{inference_type} not supported')
        search = getattr(ott,inference_type)
        return search(self, batch,
                    bos_idx, eos_idx, **kwargs)

    def infer_to_sentence(self, batch: Batch,
                bos_idx, eos_idx, inference_type: str = 'beam_search', **kwargs):
        tgts = self.infer(batch, bos_idx, eos_idx, inference_type, **kwargs)
        tgts = tgts.cpu().tolist()
        comments = []
        for sentence in tgts:
            comments.append(' '.join(batch.vocab.lookup_tokens(sentence)))
        return self.outprocess(comments)

    def outprocess(self,sentences):
        return [sent.split('<eos>')[0] for sent in sentences]

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    @staticmethod
    def metrics(sentences, golds):
        BLEU = bleu_score_(sentences,golds)
        return BLEU