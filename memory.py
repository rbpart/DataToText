#%%
import torch.nn as nn
import torch
from model.parser import HyperParameters
from model.build_model import build_model
from model.dataset import IDLDataset
from model.trainer import Trainer
from torch.utils.tensorboard.writer import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

if __name__=="__main__":
    writer = SummaryWriter()
    hparameters = HyperParameters()
    hparameters.device = 'cpu'

    dataset = IDLDataset(hparameters, 'train')
    test_dataset = IDLDataset(hparameters, 'test')

    model = build_model(hparameters,dataset)
    src, src_len, src_map, tar, tar_lengths, tokens, fusedvocab = dataset[[0,1]]
    # writer.add_graph(model,(src,tar,src_len,src_map))

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tgt_vocab([dataset.tgt_pad_word])[0],
                        reduction="sum")
    optim = torch.optim.Adam(model.parameters(), lr=hparameters.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim,[lambda epoch: 0.95**(epoch/100)])

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True, with_stack=True) as prof:
        with record_function("forward"):
            out = model(src,tar,src_len,src_map)
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_memory_usage", row_limit=10))


# %%
