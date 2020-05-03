
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

def display_attention(candidate,translation,attention):
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111)
    attention= attention.squeeze(1).cpu().detach().numpy()
    cax=ax.matshow(attention,cmap='bone')
    ax.tick_params(labelsize=15)
    ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in (candidate.split(' '))]+['<eos>'])
    ax.set_yticklabels(['']+translation.split(' '))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig('./test2.jpg')
    plt.show()
    plt.close()


src = "zwei kleinkinder im freien auf dem gras."
trg = "two people"
attention = torch.tensor([[0.1,0.14,0.28,0.17,0.36,0.67,0.5,0.22,0.01],[0.1,0.14,0.28,0.17,0.36,0.67,0.5,0.22,0.01]])
print(attention.shape)
display_attention(src, trg, attention)


def showgraph(attn):
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels(['']+sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels(['']+sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()



