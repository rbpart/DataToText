from torch import nn as nn
import torch
from decoders.hierarchical_decoder import HierarchicalRNNDecoder
from encoders.hierarchical_transformer import HierarchicalTransformerEncoder
from tqdm import tqdm


class DataToTextModel(nn.Module):
    def __init__(self, encoder: HierarchicalTransformerEncoder, decoder: HierarchicalRNNDecoder, generator: nn.Sequential,
        dataset_model = None, device = None) -> None:
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

    def forward(self, src, tgt = None, lengths = None, bptt=False, with_align=False):
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
        enc_state, memory_bank, lengths = self.encoder(src, lengths = None)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                    memory_lengths=lengths,
                                    with_align=with_align)

        return self.generator(dec_out).transpose(0,1)

    #Utilitaries for the model

    def _select_dataset(self,dataset):
        if dataset is not None:
            dataset = dataset
        elif self.dataset is not None:
            dataset = self.dataset
        else:
            raise RuntimeError('Provide the dataset with the vocab used for training or initialize model with the dataset')
        return dataset

    def infer(self,src,dataset = None):
        dataset = self._select_dataset(dataset)
        _,batch_size,_ = src.shape
        beggining = [dataset.tgt_vocab(t) for t in [['<bos>']*batch_size]]
        tgt = torch.zeros((dataset.max_comment_len,batch_size),dtype=torch.long, device=self.device, requires_grad = False)
        tgt[0,:] = torch.tensor(beggining, device=self.device, requires_grad = False)
        self.eval()
        with torch.no_grad():
            for i in tqdm(range(1,dataset.max_comment_len)):
                out = self.forward(src,tgt[:i,:])
                tgt[i,:] = out.argmax(2).transpose(0,1)[i-1,:]
        self.train()
        return tgt

    def infer_to_sentence(self, src, dataset=None):
        dataset = self._select_dataset(dataset)
        tgts = self.infer(src,dataset).transpose(0,1).tolist()
        comments = []
        for sentence in tgts:
            comments.append(' '.join(dataset.tgt_vocab.lookup_tokens(sentence)))
        return comments

    def metrics(sentences, golds):
        return