import logging
import torch
import s2s
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
import os

try:
    import ipdb
except ImportError:
    pass

lower = True
seq_length = 100
report_every = 100000
shuffle = 1

logger = logging.getLogger(__name__)


def makeVocabulary(filenames, size):
    vocab = s2s.Dict([s2s.Constants.PAD_WORD, s2s.Constants.UNK_WORD,
                      s2s.Constants.BOS_WORD, s2s.Constants.EOS_WORD], lower=lower)
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            for sent in f.readlines():
                for word in sent.strip().split(' '):
                    vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    logger.info('Created dictionary of size %d (pruned from %d)' %
                (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFiles, vocabFile, vocabSize):
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        logger.info('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = s2s.Dict(lower=lower)
        vocab.loadFile(vocabFile)
        logger.info('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        logger.info('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFiles, vocabSize)

        vocab = genWordVocab

    return vocab


def saveVocabulary(name, vocab, file):
    logger.info('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, featFiles, tgtFile, srcDicts,  featDicts, tgtDicts):
    src, tgt, ori_src, ori_tgt = [], [], [], []
    feats = []
    switch, c_tgt = [], []
    sizes = []
    count, ignored = 0, 0

    logger.info('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile, encoding='utf-8')
    tgtF = open(tgtFile, encoding='utf-8')
    featFs = [open(x, encoding='utf-8') for x in featFiles]

    index = 0
    while True:
        index += 1
        # if index % 10000 == 0: print("preprocess:"+str(index))
        if index == 10241:break
        sline = srcF.readline()
        tline = tgtF.readline()
        featLines = [x.readline() for x in featFs]

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            logger.info('WARNING: source and target do not have the same number of sentences')
            break

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
            ori_src += [srcWords]
            ori_tgt += [tgtWords]
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
    new_tgt = {}
    new_ori_src = []
    new_ori_tgt = []
    #tgts = {}
    tgts = []
    #switchs = {}
    switchs = []
    #c_tgts = {}
    c_tgts = []
    #ori_tgts = {}
    ori_tgts = []
    new_feats = []
    new_switch =[]
    new_c_tgt = []
    new_sizes = []
    store_map = {}
    #print("srclen:"+str(len(src))+":"+str(src[0])+":"+str(src[1]))

    for i in range(count):
        # if i % 1000 == 0: print("processing:"+str(i))
        ismatch = False
        index_new = -1
        for index_new, new_src_item in enumerate(new_src):
            if src[i].equal(new_src_item):
               ismatch = True
               break
        if not ismatch:
        #if src[i] not in new_src:
            new_src.append(src[i])
            new_ori_src.append(ori_src[i])
            #tgts[src[i]] = [tgt[i]]
            tgts.append([tgt[i]])
            #ori_tgts[src[i]] = [ori_tgt[i]]
            ori_tgts.append([ori_tgt[i]])
            #switchs[src[i]] = [switch[i]]
            switchs.append([switch[i]])
            #c_tgts[src[i]] = [c_tgt[i]]
            c_tgts.append([c_tgt[i]])
            new_sizes.append(sizes[i])
            new_feats.append(feats[i])
        else:
            #print('dd:', len(new_src), [src[i]] in new_src)
            tgts[index_new].append(tgt[i])
            switchs[index_new].append(switch[i])
            c_tgts[index_new].append(c_tgt[i])
            ori_tgts[index_new].append(ori_tgt[i])
    if shuffle == 1:
        logger.info('... shuffling sentences')
        perm = torch.randperm(len(src))
        new_src = [new_src[idx] for idx in perm]
        new_ori_src = [new_ori_src[idx] for idx in perm]
        new_tgt = [tgts[idx] for idx in perm]
        new_ori_tgt = [ori_tgts[idx] for idx in perm]
        new_switch = [switchs[idx] for idx in perm]
        new_c_tgt = [c_tgts[idx] for idx in perm]
        #for src in new_src:
        #    new_tgt.append(tgts[src])
        #    new_ori_tgt.append(ori_tgts[src])
        #    new_switch.append(switchs[src])
        #    new_c_tgt.append(c_tgts[src])
        new_feats = [new_feats[idx] for idx in perm]
        new_sizes = [new_sizes[idx] for idx in perm]

    logger.info('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(new_sizes))
    new_src = [new_src[idx] for idx in perm]
    new_ori_src = [new_ori_src[idx] for idx in perm]
    new_tgt = [tgts[idx] for idx in perm]
    new_ori_tgt = [ori_tgts[idx] for idx in perm]
    new_switch = [switchs[idx] for idx in perm]
    new_c_tgt = [c_tgts[idx] for idx in perm]
    #for src in new_src:
    #    new_tgt.append(tgts[src])
    #    new_ori_tgt.append(ori_tgts[src])
    #    new_switch.append(switchs[src])
    #    new_c_tgt.append(c_tgts[src])
    new_feats = [new_feats[idx] for idx in perm]

    logger.info('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
                (len(new_src), ignored, seq_length))
    return new_src, new_feats, new_tgt, new_switch, new_c_tgt, new_ori_src, new_ori_tgt

def save_vocab_embed(glove_path, dicts):
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format(glove_path, binary=False, encoding='utf-8')
    weight = torch.zeros(dicts.size(), 300)
    for i in range(len(wvmodel.index2word)):
        try:
            index = dicts.labelToIdx[wvmodel.index2word[i]]
        except:
            continue
        weight[index, :] = torch.from_numpy(wvmodel.get_vector(dicts.idxToLabel[dicts.labelToIdx[wvmodel.index2word[i]]]))

    return weight

def prepare_data_online(train_src, src_vocab, train_feats, feat_vocab, train_tgt, tgt_vocab,glove_path):
    dicts = {}
    dicts['src'] = initVocabulary('source', [train_src], src_vocab, 0)
    dicts['feat'] = initVocabulary('feat', [train_feats], feat_vocab, 0)
    dicts['tgt'] = initVocabulary('target', [train_tgt], tgt_vocab, 0)
    if glove_path:
        emb_weight = save_vocab_embed(glove_path,dicts['src'])
    else:
        emb_weight = None
    logger.info('Preparing training ...')
    train = {}
    train['src'],  train['feats'], \
    train['tgt'], train['switch'], train['c_tgt'],train['ori_src'],train['ori_tgt'] = makeData(train_src,  train_feats,
                                                             train_tgt,dicts['src'], dicts['feat'],dicts['tgt'])

    dataset = {'dicts': dicts,
               'train': train,
               # 'valid': valid
               'emb_weight': emb_weight,
               }
    return dataset
