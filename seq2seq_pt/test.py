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

p= torch.tensor([0.5,0.1,0.4])

# p = F.softmax(p, dim=-1)
q= torch.tensor([0.4,0.4,0.2])

# q =F.softmax(q, dim=-1)
print(q)
# _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
#                                - F.log_softmax(q_logit, dim=-1)), 1)
fun = nn.KLDivLoss(size_average=False, reduce=True)
kl = fun(p,q)

kl2 = F.kl_div(q.log(), p,  reduction='sum')
kl22 = F.kl_div(q.log(), p, size_average=True, reduce=True, reduction='sum')
kl_3 = (p* F.log_softmax(p/q)).sum()
print(kl,kl2,kl22,kl_3)
'''
# a = torch.randn(16,8,30)
#
# b =torch.split(a,1,1)
# print(len(b))
# print(b[0].shape)
'''
'''
multi = []
for i in range(17):
    a = torch.randn([7, 8, 77])
    multi.append(a)
mulattn=[]

print(len(multi),multi[0].shape)
print(multi[0][0])
for j in range(16, -1, -1):
    print(j)
    mulattn.append(multi[j][0])
c= torch.stack(mulattn[::-1])
#[sen_len, n_heads, src_len]
print(c.shape)
'''

