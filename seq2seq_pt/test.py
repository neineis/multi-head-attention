import  torch

import  torch.nn as nn
import torch.nn.functional as F

# p_logit: [batch, class_num]
# q_logit: [batch, class_num]
tmps = []
'''
for i in range(20):
    tmp= []
    for j in range(8):
        p_logit = torch.randn(32,16)

        p = F.softmax(p_logit,dim=-1)
        tmp += [p]

    tmps.append(tmp)
print(len(tmps),len(tmp),tmps[0][0].shape)

print(kl_categorical(tmps))
    # return _kl

# print(kl_categorical(p,q))
#py
_
# print(_kl2)
'''

p= torch.tensor([0.5,0.1,0.4,0])

# p = F.softmax(p, dim=-1)
q= torch.tensor([0.4,0.4,0.2])

flag = p.data.eq(0).float()
print(1-flag)
# a = torch.randn(16,8,30)
#
# b =torch.split(a,1,1)
# print(len(b))
# print(b[0].shape)


