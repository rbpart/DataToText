#%%
from model.parser import HyperParameters
from model.build_model import build_model
from model.dataset import IDLDataset

if __name__=="__main__":
    hparameters = HyperParameters()

    dataset = IDLDataset(hparameters, 'train')
    test_dataset = IDLDataset(hparameters, 'test')

    model = build_model(hparameters,dataset)
    src, src_len, src_map, tar, tar_lengths, tokens, fusedvocab = dataset[[0,1]]
    outputs = model.infer_to_sentence(src, src_len, src_map,fusedvocab,dataset)

# %%

from model.utils import bleu_score_

score = bleu_score_(outputs,tokens)

# %%
