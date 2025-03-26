import torch

pth = r'resnet18-5c106cde.pth'
sta_dic = torch.load(pth)
print('.pth type:', type(sta_dic))
print('.pth len:', len(sta_dic))
print('--------------------------')
for k in sta_dic.keys():
    print(k, type(sta_dic[k]), sta_dic[k].shape)