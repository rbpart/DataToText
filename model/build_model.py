from model.model import DataToTextModel
from modules.table_embeddings import TableEmbeddings
from decoders.hierarchical_decoder import HierarchicalRNNDecoder
from encoders.hierarchical_transformer import HierarchicalTransformerEncoder
import torch.nn as nn
import torch
from model.dataset import IDLDataset
from torch.nn import Embedding
from model.parser import Opts

def build_embeddings(opt: Opts, loader: IDLDataset, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    if for_encoder:
        emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size

        word_vocab_size = len(loader.src_vocab)
        word_padding_idx = loader.src_vocab([loader.src_pad_word])[0]

        feat_vocab_size = len(loader.src_vocab_feat)
        feat_padding_idx = loader.src_vocab([loader.src_pad_feat])[0]

        ent_idx = None # loader.src_word_vocab['<ent>']

        return TableEmbeddings(
            word_vec_size=emb_dim,
            word_vocab_size=word_vocab_size,
            word_padding_idx=word_padding_idx,
            feat_vec_exponent=opt.feat_vec_exponent,
            feat_vec_size=opt.feat_vec_size,
            feat_vocab_size=feat_vocab_size,
            feat_padding_idx=feat_padding_idx,
            merge=opt.feat_merge,
            merge_activation= "ReLU",#opt.feat_merge_activation,
            dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            ent_idx=ent_idx
        )

    # A refaire avec des embeddings préentrainés ici !
    emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size

    word_padding_idx = loader.tgt_vocab([loader.tgt_pad_word])[0]
    num_embs = len(loader.tgt_vocab)

    return Embedding(
        num_embeddings=num_embs,
        embedding_dim=emb_dim,
        padding_idx=word_padding_idx
    )


def build_encoder(opts,loader:IDLDataset) -> nn.Module :
    encoder_embeddings = build_embeddings(opts,loader,for_encoder=True)
    encoder = HierarchicalTransformerEncoder(
                                    embeddings=encoder_embeddings)
    return encoder

def build_decoder(opts,loader:IDLDataset) -> nn.Module :
    decoder_embeddings = build_embeddings(opts,loader,for_encoder=False)
    decoder = HierarchicalRNNDecoder(
        hidden_size=opts.rnn_size, num_layers=1, bidirectional_encoder=True,
        rnn_type="LSTM", embeddings=decoder_embeddings)
    return decoder

def build_generator(opts, loader:IDLDataset) -> nn.Module:
    return nn.Sequential(
        nn.Linear(opts.rnn_size, len(loader.tgt_vocab)),
        nn.LogSoftmax(dim=-1))

def build_model(opts: Opts, loader: IDLDataset):

    decoder = build_decoder(opts,loader)
    encoder = build_encoder(opts,loader)
    generator = build_generator(opts,loader)
    model = DataToTextModel(encoder,decoder,generator)
    model.to(opts.device)
    return model

