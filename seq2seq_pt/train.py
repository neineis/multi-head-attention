from __future__ import division

import s2s
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import datetime
import logging
import torch.nn.functional as F

try:
    import ipdb
except ImportError:
    pass
from nltk.translate import bleu_score
# from PyBLEU.eval import *

from s2s.xinit import xavier_normal, xavier_uniform
import os
import xargs

parser = argparse.ArgumentParser(description='train.py')
xargs.add_data_options(parser)
xargs.add_model_options(parser)
xargs.add_train_options(parser)

opt = parser.parse_args()

logging.basicConfig(format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
log_file_tag = time.strftime("%Y%m%d-%H%M%S")
log_file_name = log_file_tag + '.log.txt'
if opt.log_home:
    log_file_name = os.path.join(opt.log_home, log_file_name)
file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
logging.root.addHandler(file_handler)
logger = logging.getLogger(__name__)

logger.info('My PID is {0}'.format(os.getpid()))
logger.info('PyTorch version: {0}'.format(str(torch.__version__)))
logger.info(opt)

if torch.cuda.is_available() and not opt.gpus:
    logger.info("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

if opt.seed > 0:
    torch.manual_seed(opt.seed)

if opt.gpus:
    if opt.cuda_seed > 0:
        torch.cuda.manual_seed(opt.cuda_seed)
    cuda.set_device(opt.gpus[0])

logger.info('My seed is {0}'.format(torch.initial_seed()))
logger.info('My cuda seed is {0}'.format(torch.cuda.initial_seed()))


def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[s2s.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit


def loss_function(g_outputs, g_targets, generator, crit, eval=False):
    batch_size = g_outputs.size(1)

    g_out_t = g_outputs.view(-1, g_outputs.size(2))
    g_prob_t = generator(g_out_t)

    g_loss = crit(g_prob_t, g_targets.view(-1))
    total_loss = g_loss
    report_loss = total_loss.item()
    return total_loss, report_loss, 0

def diff_attn(multi_attns):
    n_heads = len(multi_attns[0])
    multi_attns = [torch.stack(x) for x in multi_attns]
    multi_attns = torch.stack(multi_attns) # len_q * n_heads * batch_size * len_v
    multi_attns = multi_attns.permute(2,1,0,3)

    x1 = multi_attns.unsqueeze(1)
    x2 = multi_attns.unsqueeze(2)

    x1 =x1.repeat(1,n_heads,1,1,1)
    x2= x2.repeat(1,1,n_heads,1,1)
    mul_diff = torch.mul(x1,x2)
    mul_diff = mul_diff.permute(0, 3, 1, 2, 4)
    mul_diff = (mul_diff.sum(-1).sum(-1).sum(-1).sum())/(n_heads*n_heads)
    return  mul_diff


def kl_categorical(tmps):

    kls = 0
    for i in range (len(tmps)):
        for j in range(len(tmps[0])-1):
            for k in range(i+1,len(tmps[0])):
                kls += F.kl_div(tmps[i][k].log(), tmps[i][j],size_average=True)

    return kls

def generate_copy_loss_function(g_outputs, c_outputs, g_targets,
                                c_switch, c_targets, c_gate_values,
                                generator, crit, copyCrit,mul_head_attns):
    batch_size = g_outputs.size(1)

    g_out_t = g_outputs.view(-1, g_outputs.size(2))
    g_prob_t = generator(g_out_t)
    g_prob_t = g_prob_t.view(-1, batch_size, g_prob_t.size(1))

    c_output_prob = c_outputs * c_gate_values.expand_as(c_outputs) + 1e-8
    g_output_prob = g_prob_t * (1 - c_gate_values).expand_as(g_prob_t) + 1e-8

    c_output_prob_log = torch.log(c_output_prob)
    g_output_prob_log = torch.log(g_output_prob)
    c_output_prob_log = c_output_prob_log * (c_switch.unsqueeze(2).expand_as(c_output_prob_log))
    g_output_prob_log = g_output_prob_log * ((1 - c_switch).unsqueeze(2).expand_as(g_output_prob_log))

    g_output_prob_log = g_output_prob_log.view(-1, g_output_prob_log.size(2))
    c_output_prob_log = c_output_prob_log.view(-1, c_output_prob_log.size(2))

    g_loss = crit(g_output_prob_log, g_targets.view(-1))
    c_loss = copyCrit(c_output_prob_log, c_targets.view(-1))
    kl_loss = 0
    # kl_loss = diff_attn(mul_head_attns)
    # total_loss = g_loss + c_loss + kl_loss * 100
    total_loss = g_loss + c_loss
    # print(torch.isnan(total_loss),torch.isnan(kl_loss),total_loss.item(),kl_loss.item())
    report_loss = total_loss.item()
    # print(report_loss)
    return total_loss, report_loss, kl_loss, 0


def addPair(f1, f2):
    for x, y1 in zip(f1, f2):
        yield (x, y1)
    yield (None, None)


def load_dev_data(translator, src_file,  feat_files, tgt_file):
    dataset, raw = [], []
    srcF = open(src_file, encoding='utf-8')
    tgtF = open(tgt_file, encoding='utf-8')
    featFs = [open(x, encoding='utf-8') for x in feat_files]

    src_batch, tgt_batch = [], []
    feats_batch = []
    for line, tgt in addPair(srcF, tgtF):
        if (line is not None) and (tgt is not None):
            src_tokens = line.strip().split(' ')
            src_batch += [src_tokens]
            tgt_tokens = tgt.strip().split(' ')
            tgt_batch += [tgt_tokens]
            feats_tokens = [reader.readline().strip().split((' ')) for reader in featFs]
            feats_batch += [feats_tokens]

            if len(src_batch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(src_batch) == 0:
                break
        data = translator.buildData(src_batch, feats_batch, tgt_batch)
        dataset.append(data)
        raw.append((src_batch, tgt_batch))
        src_batch, tgt_batch = [], []
        feats_batch = []
    srcF.close()
    for f in featFs:
        f.close()
    tgtF.close()
    return (dataset, raw)


evalModelCount = 0
totalBatchCount = 0


def evalModel(model, translator, evalData):
    global evalModelCount
    evalModelCount += 1
    ofn = 'dev.out.{0}'.format(evalModelCount)
    srcfn = 'dev.src.{0}'.format(evalModelCount)
    tgtfn = 'dev.tgt.{0}'.format(evalModelCount)
    file_name_tag = 'result_' + log_file_tag
    if opt.save_path:
        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        save_path_tmp = opt.save_path + os.path.sep + file_name_tag
        if not os.path.exists(save_path_tmp):
            os.makedirs(save_path_tmp)
        ofn = os.path.join(save_path_tmp, ofn)
        srcfn =os.path.join(save_path_tmp, srcfn)
        tgtfn =os.path.join(save_path_tmp, tgtfn)

    predict, gold ,passages= [], [],[]
    processed_data, raw_data = evalData
    for batch, raw_batch in zip(processed_data, raw_data):
        """
        (wrap(srcBatch), lengths), \
               (wrap(bioBatch), lengths), (tuple(wrap(x) for x in featBatches), lengths), \
               (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch)), \
               indices
        """
        src, feats, tgt, indices = batch[0]
        src_batch, tgt_batch = raw_batch

        #  (2) translate
        pred, predScore, predIsCopy, predCopyPosition, attn, mul_attn, _ = translator.translateBatch(src, feats, tgt)
        pred, predScore, predIsCopy, predCopyPosition, attn = list(zip(
            *sorted(zip(pred, predScore, predIsCopy, predCopyPosition, attn, indices),
                    key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(src[0].size(1)):
            n = 0
            predBatch.append(
                translator.buildTargetTokens(pred[b][n], src_batch[b],
                                             predIsCopy[b][n], predCopyPosition[b][n], attn[b][n])
            )
        # nltk BLEU evaluator needs tokenized sentences
        gold += [[r] for r in tgt_batch]
        passages += [[r] for r in src_batch]
        predict += predBatch

    no_copy_mark_predict = [[word.replace('[[', '').replace(']]', '') for word in sent] for sent in predict]
    bleu = bleu_score.corpus_bleu(gold, no_copy_mark_predict)
    report_metric = bleu

    with open(ofn, 'w', encoding='utf-8') as of:
        for p in no_copy_mark_predict:
            of.write(' '.join(p) + '\n')
    with open(tgtfn, 'w', encoding='utf-8') as tof:
        for p in gold:
            tof.write(' '.join(p[0]) + '\n')
    with open(srcfn, 'w', encoding='utf-8') as sof:
        for p in passages:
            sof.write(' '.join(p[0]) + '\n')
    # final_results = GetEvalResult(of, tgtfn, srcfn)
    return report_metric


def trainModel(model, translator, trainData, validData, dataset, optim):
    logger.info(model)
    model.train()
    logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))

    # define criterion of each GPU
    criterion = NMTCriterion(dataset['dicts']['tgt'].size())
    copyLossF = nn.NLLLoss(size_average=False)

    start_time = time.time()

    def saveModel(metric=None):
        model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
        generator_state_dict = model.generator.module.state_dict() if len(
            opt.gpus) > 1 else model.generator.state_dict()
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optim': optim
        }

        save_model_path = 'model'
        file_name_tag = 'result_' + log_file_tag
        if opt.save_path:
            if not os.path.exists(opt.save_path):
                os.makedirs(opt.save_path)
            save_path_tmp = opt.save_path + os.path.sep + file_name_tag
            if not os.path.exists(save_path_tmp):
                os.makedirs(save_path_tmp)
            save_model_path = save_path_tmp + os.path.sep + save_model_path
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
        if metric is not None:
            torch.save(checkpoint, '{0}_dev_metric_{1}_e{2}.pt'.format(save_model_path, round(metric, 4), epoch))
        else:
            torch.save(checkpoint, '{0}_e{1}.pt'.format(save_model_path, epoch))

    def trainEpoch(epoch):

        if opt.extra_shuffle and epoch > opt.curriculum:
            logger.info('Shuffling...')
            trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_loss, total_kl_loss, total_words, total_num_correct = 0, 0, 0,0
        report_loss, report_tgt_words, report_src_words, report_num_correct = 0, 0, 0, 0
        start = time.time()
        for i in range(len(trainData)):
            global totalBatchCount
            totalBatchCount += 1
            """
            (wrap(srcBatch), lengths), \
               (wrap(bioBatch), lengths), ((wrap(x) for x in featBatches), lengths), \
               (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch)), \
               indices
            """
            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx][:-1]  # exclude original indices

            model.zero_grad()
            # ipdb.set_trace()

            g_outputs, c_outputs, c_gate_values, mul_head_attns = model(batch)
            targets = batch[2][0][1:]  # exclude <s> from targets
            copy_switch = batch[2][1][1:]
            c_targets = batch[2][2][1:]
            #print('mul_head_attns:',len(mul_head_attns),len(mul_head_attns[0]),mul_head_attns[0][0].shape)
            # loss, res_loss, num_correct = loss_function(g_outputs, targets, model.generator, criterion)
            loss, res_loss, kl_loss, num_correct = generate_copy_loss_function(
                g_outputs, c_outputs, targets, copy_switch, c_targets, c_gate_values, model.generator, criterion,
                copyLossF, mul_head_attns)

            if math.isnan(res_loss) or res_loss > 1e20:
                if math.isnan(res_loss):
                    logger.info('catch NaN')
                else:
                    logger.info('res_loss > 1e20')
                ipdb.set_trace()
            # update the parameters
            loss.backward()
            optim.step()

            num_words = targets.data.ne(s2s.Constants.PAD).sum().item()
            report_loss += res_loss
            report_num_correct += num_correct
            report_tgt_words += num_words
            report_src_words += batch[0][-1].data.sum()
            total_loss += res_loss
            total_kl_loss += kl_loss
            total_num_correct += num_correct
            total_words += num_words
            if i % opt.log_interval == -1 % opt.log_interval:
                logger.info(
                    "Epoch %2d, %6d/%5d/%5d; acc: %6.2f; loss: %6.2f; kl_loss:%6.2f; words: %5d; ppl: %6.2f; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed" %
                    (epoch, totalBatchCount, i + 1, len(trainData),
                     report_num_correct / report_tgt_words * 100,
                     report_loss,
                     kl_loss,
                     report_tgt_words,
                     math.exp(min((report_loss / report_tgt_words), 16)),
                     report_src_words / max((time.time() - start), 1.0),
                     report_tgt_words / max((time.time() - start), 1.0),
                     time.time() - start))

                report_loss = report_tgt_words = report_src_words = report_num_correct = 0
                start = time.time()

            if validData is not None and totalBatchCount % opt.eval_per_batch == -1 % opt.eval_per_batch \
                    and totalBatchCount >= opt.start_eval_batch:
                model.eval()
                logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))
                valid_bleu= evalModel(model, translator, validData)
                model.train()
                logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))
                model.decoder.attn.mask = None
                logger.info('Validation Score: %g' % (valid_bleu * 100))
                if valid_bleu >= optim.best_metric:
                    saveModel(valid_bleu)
                optim.updateLearningRate(valid_bleu, epoch)

        return total_loss / total_words, total_num_correct / total_words
        # return 0, 0

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        logger.info('')
        #  (1) train for one epoch on the training set
        train_loss, train_acc = trainEpoch(epoch)
        train_ppl = math.exp(min(train_loss, 100))
        logger.info('Train perplexity: %g' % train_ppl)
        logger.info('Train accuracy: %g' % (train_acc * 100))
        logger.info('Saving checkpoint for epoch {0}...'.format(epoch))
        saveModel()


def main():
    import onlinePreprocess
    opt = parser.parse_args()
    onlinePreprocess.lower = opt.lower_input
    onlinePreprocess.seq_length = opt.max_sent_length
    onlinePreprocess.shuffle = 1 if opt.process_shuffle else 0
    from onlinePreprocess import prepare_data_online
    dataset = prepare_data_online(opt.train_src, opt.src_vocab, opt.train_feats,
                                  opt.feat_vocab, opt.train_tgt, opt.tgt_vocab, opt.glove_path)

    trainData = s2s.Dataset(dataset['train']['src'], dataset['train']['feats'],
                            dataset['train']['tgt'],
                            dataset['train']['switch'], dataset['train']['c_tgt'],
                            opt.batch_size, opt.gpus)
    dicts = dataset['dicts']
    emb_weight = dataset['emb_weight']
    logger.info(' * vocabulary size. source = %d; target = %d' %
                (dicts['src'].size(), dicts['tgt'].size()))
    logger.info(' * number of training sentences. %d' %
                len(dataset['train']['src']))
    logger.info(' * maximum batch size. %d' % opt.batch_size)

    logger.info('Building model...')

    if True:
        encoder = s2s.Models.Encoder(opt, dicts['src'], emb_weight)
        decoder = s2s.Models.Decoder(opt, dicts['tgt'], emb_weight)
        decIniter = s2s.Models.DecInit(opt)

        generator = nn.Sequential(
            nn.Linear(opt.dec_rnn_size // opt.maxout_pool_size, dicts['tgt'].size()),  # TODO: fix here
            # nn.LogSoftmax(dim=1)
            nn.Softmax(dim=1)
        )

        model = s2s.Models.NMTModel(encoder, decoder, decIniter)
        model.generator = generator
        translator = s2s.Translator(opt, model, dataset)

        if len(opt.gpus) >= 1:
            model.cuda()
            generator.cuda()
        else:
            model.cpu()
            generator.cpu()

        # if len(opt.gpus) > 1:
        #    model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
        #    generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)

        for pr_name, p in model.named_parameters():
            logger.info(pr_name)
            # p.data.uniform_(-opt.param_init, opt.param_init)
            if p.dim() == 1:
                # p.data.zero_()
                p.data.normal_(0, math.sqrt(6 / (1 + p.size(0))))
            else:
                nn.init.xavier_normal_(p, math.sqrt(3))

        # encoder.load_pretrained_vectors(opt)
        # decoder.load_pretrained_vectors(opt)

        optim = s2s.Optim(
            opt.optim, opt.learning_rate,
            max_grad_norm=opt.max_grad_norm,
            max_weight_value=opt.max_weight_value,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            decay_bad_count=opt.halve_lr_bad_count
        )
        optim.set_parameters(model.parameters())
    else:

        checkpoint = torch.load(opt.train_from)
        opt = checkpoint['opt']
        src_dict = checkpoint['dicts']['src']
        tgt_dict = checkpoint['dicts']['tgt']
        weight = None
        encoder = s2s.Models.Encoder(opt, src_dict, weight)
        decoder = s2s.Models.Decoder(opt, tgt_dict, weight)
        decIniter = s2s.Models.DecInit(opt)
        model = s2s.Models.NMTModel(encoder, decoder, decIniter)

        generator = nn.Sequential(
            nn.Linear(opt.dec_rnn_size // opt.maxout_pool_size, tgt_dict.size()),
            nn.Softmax())  # TODO pay attention here
        optim = s2s.Optim(
            opt.optim, opt.learning_rate,
            max_grad_norm=opt.max_grad_norm,
            max_weight_value=opt.max_weight_value,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            decay_bad_count=opt.halve_lr_bad_count
        )
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
        model.generator = generator
        optim.load_state_dict(checkpoint['optim'])
        if len(opt.gpus) >= 1:
            model.cuda()
            generator.cuda()
        else:
            model.cpu()
            generator.cpu()


    validData = None
    if opt.dev_input_src and opt.dev_ref:
        validData = load_dev_data(translator, opt.dev_input_src,  opt.dev_feats, opt.dev_ref)
    trainModel(model, translator, trainData, validData, dataset, optim)


if __name__ == "__main__":
    main()
