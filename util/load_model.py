import torch
model_path = '/research/dept7/glchen/github/pixel2pixel/checkpoints/dcupp_naive6_50epoch/50_net_G.pth'
model_dict = torch.load(model_path)
dict_name = list(model_dict)
for i, p in enumerate(dict_name):
    print(i, p)