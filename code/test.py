import torch

ts = torch.empty(1,0,3)
ts = ts.unsqueeze(1)
print(ts.shape)
print(ts.shape[2])
