from __future__ import division

import s2s
import torch
import torch.nn as nn
import argparse
import math
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.pyplot import MultipleLocator


logging.basicConfig(format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
file_handler = logging.FileHandler(time.strftime("%Y%m%d-%H%M%S") + '.log.txt', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
logging.root.addHandler(file_handler)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src', required=True,
                    help='Source sequence to decode (one line per sequence)')

parser.add_argument('-feats', default=[], nargs='+', type=str)
parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-output', default='/search/odin/zll/NQG/data/models/NQG_plus/result_0429_125215/pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size', type=int, default=8,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=1,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had the highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
parser.add_argument('-verbose', action="store_true",
                    help='logger.info scores and predictions for each sentence')
parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")

parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")


def reportScore(name, scoreTotal, wordsTotal):
    logger.info("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal / wordsTotal)))


def addone(f):
    for line in f:
        yield line
    yield None


def addPair(f1, f2):
    for x, y1 in zip(f1, f2):
        yield (x, y1)
    yield (None, None)

def display_attention(candidate,translation,attention,attn_ofn):
    fig=plt.figure(figsize=(20,20))
    ax=fig.add_subplot(111)
    attention= attention.squeeze(1).cpu().detach().numpy()
    cax=ax.matshow(attention,cmap='bone')
    ax.tick_params(labelsize=15)
    ax.tick_params(width=4, colors='gold')
    ax.tick_params(length=10, colors='gold')
    ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in (candidate.split(' '))]+['<eos>'])
    ax.set_yticklabels(['']+translation.split(' '))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(attn_ofn)
    plt.show()
    plt.close()

def showAtt():
    attn_file = '/search/odin/zll/NQG/data/models/NQG_plus/attentions/'
    srcBatch = [['after', 'the', 'death', 'of', 'tugh', 'temür', 'in', '1332', 'and', 'subsequent', 'death', 'of', 'rinchinbal', '-lrb-', 'emperor', 'ningzong', '-rrb-', 'the', 'same', 'year', ',', 'the', '13-year-old', 'toghun', 'temür', '-lrb-', 'emperor', 'huizong', '-rrb-', ',', 'the', 'last', 'of', 'the', 'nine', 'successors', 'of', 'kublai', 'khan', ',', 'was', 'summoned', 'back', 'from', 'guangxi', 'and', 'succeeded', 'to', 'the', 'throne', '.']]
    feats_batch =[[['IN', 'DT', 'NN', 'IN', 'NNP', 'NNP', 'IN', 'CD', 'CC', 'JJ', 'NN', 'IN', 'NNP', '-LRB-', 'NNP', 'NNP', '-RRB-', 'DT', 'JJ', 'NN', ',', 'DT', 'JJ', 'NNP', 'NNP', '-LRB-', 'NNP', 'NNP', '-RRB-', ',', 'DT', 'JJ', 'IN', 'DT', 'CD', 'NNS', 'IN', 'NNP', 'NNP', ',', 'VBD', 'VBN', 'RB', 'IN', 'NNP', 'CC', 'VBD', 'TO', 'DT', 'NN', '.'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'DATE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'PERSON', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'NUMBER', 'O', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'LOCATION', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['UP', 'LOW', 'LOW', 'LOW', 'UP', 'UP', 'LOW', 'LOW', 'LOW', 'LOW', 'LOW', 'LOW', 'UP', 'LOW', 'UP', 'UP', 'LOW', 'LOW', 'LOW', 'LOW', 'LOW', 'LOW', 'LOW', 'UP', 'UP', 'LOW', 'UP', 'UP', 'LOW', 'LOW', 'LOW', 'LOW', 'LOW', 'LOW', 'LOW', 'LOW', 'LOW', 'UP', 'UP', 'LOW', 'LOW', 'LOW', 'LOW', 'LOW', 'UP', 'LOW', 'LOW', 'LOW', 'LOW', 'LOW', 'LOW']]]
    tgtBatch =[['when', 'did', 'tugh', 'temur', 'die', '?\n']]
    #[[['what', 'was', 'the', 'last', 'year', 'of', '<unk>', 'khan', '?']]]
    opt = parser.parse_args()
    logger.info(opt)
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = s2s.Translator(opt)
    predBatch, predScore, mul_attn,goldScore = translator.translate(srcBatch, feats_batch, tgtBatch)
    mul_attns = torch.split(mul_attn[0],1,1)
    for i in range(len(mul_attns)):
        attn_ofn = attn_file + 'head_'+str(i)+'.jpg'
        attn = mul_attns[i].squeeze(1)
        src = ' '.join(srcBatch[0])
        tgt = ' '.join(tgtBatch[0])
        out = ' '.join(predBatch[0][0])
        display_attention(src,out,attn,attn_ofn)
    print(len(mul_attns))
    print(mul_attns[0].shape)
    # print(predBatch,mul_attn,len(mul_attn),mul_attn[0].shape)

def main():
    opt = parser.parse_args()
    logger.info(opt)
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = s2s.Translator(opt)

    outF = open(opt.output, 'w', encoding='utf-8')

    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

    srcBatch, tgtBatch = [], []
    feats_batch =  []

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None
    featFs = [open(x, encoding='utf-8') for x in opt.feats]
    for line in addone(open(opt.src, encoding='utf-8')):

        if (line is not None):
            srcTokens = line.strip().split(' ')
            srcBatch += [srcTokens]
            feats_tokens = [reader.readline().strip().split((' ')) for reader in featFs]
            feats_batch += [feats_tokens]
            if tgtF:
                tgtTokens = tgtF.readline().split(' ') if tgtF else None
                tgtBatch += [tgtTokens]

            if len(srcBatch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break
        predBatch, predScore, mul_attn, goldScore = translator.translate(srcBatch,feats_batch, tgtBatch)

        predScoreTotal += sum(score[0] for score in predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)

        # if tgtF is not None:
        #     goldScoreTotal += sum(goldScore)
        #     goldWordsTotal += sum(len(x) for x in tgtBatch)

        for b in range(len(predBatch)):
            count += 1

            sent = [word.replace('[[', '').replace(']]', '') for word in predBatch[b][0]]
            outF.write(" ".join(sent) + '\n')
            outF.flush()

            if opt.verbose:
                srcSent = ' '.join(srcBatch[b])
                if translator.tgt_dict.lower:
                    srcSent = srcSent.lower()
                logger.info('SENT %d: %s' % (count, srcSent))
                logger.info('PRED %d: %s' % (count, " ".join(predBatch[b][0])))
                logger.info("PRED SCORE: %.4f" % predScore[b][0])

                if tgtF is not None:
                    tgtSent = ' '.join(tgtBatch[b])
                    if translator.tgt_dict.lower:
                        tgtSent = tgtSent.lower()
                    logger.info('GOLD %d: %s ' % (count, tgtSent))
                    # logger.info("GOLD SCORE: %.4f" % goldScore[b])

                if opt.n_best > 1:
                    logger.info('\nBEST HYP:')
                    for n in range(opt.n_best):
                        logger.info("[%.4f] %s" % (predScore[b][n], " ".join(predBatch[b][n])))

                logger.info('')

        srcBatch, tgtBatch = [], []
        feats_batch = []

    reportScore('PRED', predScoreTotal, predWordsTotal)

    if tgtF:
        tgtF.close()

    logger.info('{0} copy'.format(translator.copyCount))


if __name__ == "__main__":
    showAtt()
