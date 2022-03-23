#%%
from model.parser import HyperParameters
from model.build_model import build_model
from model.dataset import IDLDataset
from model.utils import bleu_score_
from model.metrics import metrics

if __name__=="__main__":
    hparameters = HyperParameters()
    hparameters.device = 'cuda'
    dataset = IDLDataset(hparameters, 'train')
    test_dataset = IDLDataset(hparameters, 'test')
    model = build_model(hparameters,dataset)
    model.load_params('model/models/experiment7/experiment7checkpoint_7500/model_7500.pt')
    batch = dataset[[0,2]]
    outputs = model(batch)
    outputs = model.infer_to_sentence(batch,2,predictions=100,beam_width=10)
    score = bleu_score_(outputs,batch.target.raw)
    print(score)
#scores = metrics(outputs,batch.target.raw,batch.source.raw)

# %%
