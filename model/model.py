from torch import nn as nn
import torch
from decoders.hierarchical_decoder import HierarchicalRNNDecoder
from encoders.hierarchical_transformer import HierarchicalTransformerEncoder

class DataToTextModel(nn.Module):
    def __init__(self, encoder: HierarchicalTransformerEncoder, decoder: HierarchicalRNNDecoder, generator: nn.Sequential) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

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

        if tgt is not None:
            if bptt is False:
                self.decoder.init_state(src, memory_bank, enc_state)
            dec_out, attns = self.decoder(tgt, memory_bank,
                                        memory_lengths=lengths,
                                        with_align=with_align)
        else:
            tgt = torch.zeros(enc_state)
            self.decoder()
        return self.generator(dec_out).transpose(0,1)
