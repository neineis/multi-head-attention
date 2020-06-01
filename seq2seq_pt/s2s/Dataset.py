from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

import s2s



class Dataset(object):
    def __init__(self, srcData, featsData, tgtData, copySwitchData, copyTgtData,oriSrcData,oriTgtData,
                 batchSize, cuda):
        self.src = srcData
        self.feats = featsData
        if tgtData:
            self.tgt = tgtData
            # copy switch should company tgt label
            self.copySwitch = copySwitchData
            self.copyTgt = copyTgtData
            self.oriSrcData = oriSrcData
            self.oriTgtData = oriTgtData
            assert (len(self.src) == len(self.tgt))
        else:
            self.tgt = None
            self.copySwitch = None
            self.copyTgt = None
            self.oriSrcData = None
            self.oriTgtData = None
        self.device = torch.device("cuda" if cuda else "cpu")

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src) / batchSize)

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(s2s.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def _batchify_tgt(self, data, align_right=False):
        lengths = []
        y_lengths = []
        for y in data:
            lengths += [x.size(0) for x in y]
            y_lengths.append(len(y))
        max_length = max(lengths)
        max_y_length = max(y_lengths)
        out = data[0][0].new(len(data), max_y_length, max_length).fill_(s2s.Constants.PAD)

        for i in range(len(data)):
            for j in range(y_lengths[i]):
                data_length = data[i][j].size(0)
                offset = max_length - data_length if align_right else 0
                out[i][j].narrow(0, offset, data_length).copy_(data[i][j])
        return out




    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, lengths = self._batchify(
            self.src[index * self.batchSize:(index + 1) * self.batchSize],
            align_right=False, include_lengths=True)
        featBatches = [self._batchify(x[index * self.batchSize:(index + 1) * self.batchSize], align_right=False) for x
                       in zip(*self.feats)]

        if self.tgt:
            tgtBatch = self._batchify_tgt(
                self.tgt[index * self.batchSize:(index + 1) * self.batchSize])
            oriSrcBatch = self.oriSrcData[index * self.batchSize:(index + 1) * self.batchSize]
            oriTgtBatch = self.oriTgtData[index * self.batchSize:(index + 1) * self.batchSize]
        else:
            tgtBatch = None
            oriSrcBatch = None
            oriTgtBatch = None

        if self.copySwitch is not None:
            copySwitchBatch = self._batchify_tgt(
                self.copySwitch[index * self.batchSize:(index + 1) * self.batchSize])
            copyTgtBatch = self._batchify_tgt(
                self.copyTgt[index * self.batchSize:(index + 1) * self.batchSize])
        else:
            copySwitchBatch = None
            copyTgtBatch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        if tgtBatch is None:
            batch = zip(indices, srcBatch, *featBatches)
        else:
            if self.copySwitch is not None:
                batch = zip(indices, srcBatch, *featBatches, tgtBatch, copySwitchBatch, copyTgtBatch, oriSrcBatch,oriTgtBatch)
            else:
                batch = zip(indices, srcBatch,  *featBatches, tgtBatch)
        # batch = zip(indices, srcBatch) if tgtBatch is None else zip(indices, srcBatch, tgtBatch)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgtBatch is None:
            indices, srcBatch, *featBatches = zip(*batch)
        else:
            if self.copySwitch is not None:
                indices, srcBatch, *featBatches, tgtBatch, copySwitchBatch, copyTgtBatch,oriSrcBatch,oriTgtBatch = zip(*batch)
            else:
                indices, srcBatch,  *featBatches, tgtBatch = zip(*batch)
        featBatches = list(featBatches)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            b = b.to(self.device)
            return b

        def mul_wrap(b):
            if b is None:
                return b

            b = torch.stack(b, 0)
            b= b.transpose(0,2).contiguous()
            b = b.to(self.device)

            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1)

        return (wrap(srcBatch), lengths), \
             (tuple(wrap(x) for x in featBatches), lengths), \
               (mul_wrap(tgtBatch), mul_wrap(copySwitchBatch), mul_wrap(copyTgtBatch)), \
               oriSrcBatch, oriTgtBatch,\
               indices

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(
            zip(self.src,  self.feats, self.tgt, self.copySwitch, self.copyTgt,self.oriSrcData,self.oriTgtData))
        self.src,  self.feats, self.tgt, self.copySwitch, self.copyTgt,self.oriSrcData,self.oriTgtData = zip(
            *[data[i] for i in torch.randperm(len(data))])
