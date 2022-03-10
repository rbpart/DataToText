#%%
from model.parser import HyperParameters
from model.build_model import build_model
from model.dataset import IDLDataset
from model.utils import bleu_score_
from model.metrics import metrics

if __name__=="__main__":
    hparameters = HyperParameters()
    dataset = IDLDataset(hparameters, 'train')
    test_dataset = IDLDataset(hparameters, 'test')
    model = build_model(hparameters,dataset)
    model.load_params('model/models/checkpoint_3000/model_3000.pt')
    batch = dataset[[0,1,2,3]]
    outputs = model.infer_to_sentence(batch,0,predictions=100)
    score = bleu_score_(outputs,batch.target.raw)
    scores = metrics(outputs,batch.target.raw,batch.source.raw)

# %%
