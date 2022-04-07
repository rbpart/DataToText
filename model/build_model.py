#%%
from model.model import DataToTextModel
from modules.copy_generator import CopyGenerator
from modules.table_embeddings import TableEmbeddings
from decoders.hierarchical_decoder import HierarchicalRNNDecoder
from encoders.hierarchical_transformer import HierarchicalTransformerEncoder
import torch.nn as nn
from model.dataset import IDLDataset
from torch.nn import Embedding
from model.parser import HyperParameters
from torchtext.vocab import Vectors


def load_pretrained(path, dataset:IDLDataset, new_embedding: nn.Embedding): # only for target
    pretrained = Vectors(path)
    assert pretrained.dim == new_embedding.embedding_dim
    tgt_words = dataset.tgt_vocab.get_itos()
    dataset_vocab_vectors = pretrained.get_vecs_by_tokens(tgt_words,lower_case_backup=True)
    mask = (dataset_vocab_vectors != 0).any(dim=1)
    new_embedding.weight[mask].data = dataset_vocab_vectors[mask]
    return new_embedding

def build_embeddings(opt: HyperParameters, loader: IDLDataset, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
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

    embedding = Embedding(
        num_embeddings=num_embs,
        embedding_dim=emb_dim,
        padding_idx=word_padding_idx
    )
    if opt.pretrained_tgt_embeddings_path:
        embedding = load_pretrained(opt.pretrained_tgt_embeddings_path, loader, embedding)

    return embedding


def build_encoder(opts,loader:IDLDataset) -> nn.Module :
    encoder_embeddings = build_embeddings(opts,loader,for_encoder=True)
    encoder = HierarchicalTransformerEncoder(
                                    embeddings=encoder_embeddings)
    return encoder

def build_decoder(opts,loader:IDLDataset) -> nn.Module :
    decoder_embeddings = build_embeddings(opts,loader,for_encoder=False)
    decoder = HierarchicalRNNDecoder(
        hidden_size=opts.rnn_size, num_layers=2,
        coverage_attn=True,
        bidirectional_encoder=True, forcing_frequency = opts.forcing_frequency,
        rnn_type="LSTM", embeddings=decoder_embeddings, use_pos=opts.use_pos,
        dropout=opts.dropout[0] if type(opts.dropout) is list else opts.dropout)
    return decoder

def build_generator(opts, loader:IDLDataset) -> nn.Module:
    word_padding_idx = loader.tgt_vocab([loader.tgt_pad_word])[0]

    return CopyGenerator(
        opts.rnn_size,
        len(loader.tgt_vocab),
        pad_idx=word_padding_idx)
    # return nn.Sequential(
    #     nn.Linear(opts.rnn_size, len(loader.tgt_vocab)),
    #     nn.T
    #     nn.LogSoftmax(dim=-1))

def build_model(opts: HyperParameters, loader: IDLDataset):

    decoder = build_decoder(opts,loader)
    encoder = build_encoder(opts,loader)
    generator = build_generator(opts,loader)
    model = DataToTextModel(encoder,decoder,generator,device = opts.device,dataset_model = loader)
    return model


