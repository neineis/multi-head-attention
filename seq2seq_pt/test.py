import  torch
import random
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
dd_
# print(_kl2)
'''
#
# t = torch.randn(3, 6,5).cuda()  # 初始化一个tensor，从0到23，形状为（2,3,4）
# print("t--->", t)
#
# index1 = torch.randint(low=0,high=5,size=(3,1,1)) # 要选取数据的位置
# index1 = index1.repeat(1,1,5)
# print("index--->", index1)
#
# data1 = torch.gather(t,dim=1, index=index1.cuda())  # 第一个参数:从第1维挑选， 第二个参数:从该维中挑选的位置
# print("data1--->", data1,data1.shape)



# a = torch.randn(,8,30)
#
# b =torch.split(a,1,1)
# print(len(b))
# print(b[0].shape)


# weights = torch.Tensor([[0, 10, 3, 0],[1,4,6,9]])  # create a Tensor of weights
# ans = torch.multinomial(weights, 1)
# print(ans)



src, tgt = [], []
count, ignored = 0, 0
srcFile=''
tgtFile=''
srcF = open(srcFile, encoding='utf-8')
tgtF = open(tgtFile, encoding='utf-8')

while True:
    sline = srcF.readline()
    tline = tgtF.readline()
    sline = sline.strip()
    tline = tline.strip()
    featLines = [line.strip() for line in featLines]

    # source and/or target are empty
    if sline == "" or tline == "":
        logger.info('WARNING: ignoring an empty line (' + str(count + 1) + ')')
        continue

    srcWords = sline.split(' ')
    tgtWords = tline.split(' ')
    featWords = [x.split(' ') for x in featLines]

    if len(srcWords) <= seq_length and len(tgtWords) <= seq_length:
        src += [srcDicts.convertToIdx(srcWords, s2s.Constants.UNK_WORD)]
        feats += [[featDicts.convertToIdx(x, s2s.Constants.UNK_WORD) for x in featWords]]
        tgt += [tgtDicts.convertToIdx(tgtWords,
                                      s2s.Constants.UNK_WORD,
                                      s2s.Constants.BOS_WORD,
                                      s2s.Constants.EOS_WORD)]
        switch_buf = [0] * (len(tgtWords) + 2)
        c_tgt_buf = [0] * (len(tgtWords) + 2)
        for idx, tgt_word in enumerate(tgtWords):
            word_id = tgtDicts.lookup(tgt_word, None)
            if word_id is None:
                if tgt_word in srcWords:
                    copy_position = srcWords.index(tgt_word)
                    switch_buf[idx + 1] = 1
                    c_tgt_buf[idx + 1] = copy_position
        switch.append(torch.FloatTensor(switch_buf))
        c_tgt.append(torch.LongTensor(c_tgt_buf))

        sizes += [len(srcWords)]
    else:
        ignored += 1

    count += 1

    if count % report_every == 0:
        logger.info('... %d sentences prepared' % count)

srcF.close()
tgtF.close()
for x in featFs:
    x.close()
count  = len(src)
new_src = []
new_tgt = []
tgts = {}
new_feats = []
new_switch =[]
new_c_tgt = []
new_sizes = []
for i in range(count):
    if src[i] not in new_src:
        new_src.append(src[i])
        tgts[src[i]] = [tgt[i]]
        new_feats.append(feats[i])
        new_switch.append(switch[i])
        new_c_tgt.append(c_tgt[i])
        new_sizes.append(sizes[i])
    else:
        tgts[src[i]].append(tgt[i])

if shuffle == 1:
    logger.info('... shuffling sentences')
    perm = torch.randperm(len(src))
    new_src = [new_src[idx] for idx in perm]
    for src in new_src:
        new_tgt.append(tgts[src])
    new_feats = [new_feats[idx] for idx in perm]
    new_switch = [new_switch[idx] for idx in perm]
    new_c_tgt = [new_c_tgt[idx] for idx in perm]
    new_sizes = [new_sizes[idx] for idx in perm]

logger.info('... sorting sentences by size')
_, perm = torch.sort(torch.Tensor(new_sizes))
new_src = [new_src[idx] for idx in perm]
for src in new_src:
    new_tgt.append(tgts[src])
new_feats = [new_feats[idx] for idx in perm]
new_switch = [new_switch[idx] for idx in perm]
new_c_tgt = [new_c_tgt[idx] for idx in perm]

logger.info('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
            (len(new_src), ignored, seq_length))
