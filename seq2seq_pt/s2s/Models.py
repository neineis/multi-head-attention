import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import s2s.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import time

try:
    import ipdb
except ImportError:
    pass



def init_gru_wt(lstm, opt):
    for name, _ in gru.named_parameters():
        if 'weight' in name:
            wt = getattr(lstm, name)
            wt.data.uniform_(-opt.rand_unif_init_mag, opt.rand_unif_init_mag)
        elif 'bias' in name:
            # set forget bias to 1
            bias = getattr(lstm, name)
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data.fill_(0.)
            bias.data[start:end].fill_(1.)

def init_linear_wt(linear,opt):
    linear.weight.data.normal_(std=opt.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=opt.trunc_norm_init_std)

def init_wt_normal(wt,opt):
    wt.data.normal_(std=opt.trunc_norm_init_std)



class Encoder(nn.Module):
    def __init__(self, opt, dicts, weight):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.enc_rnn_size % self.num_directions == 0
        self.hidden_size = opt.enc_rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()
        if not (weight is None):
            self.word_lut = nn.Embedding.from_pretrained(weight)
            self.word_lut.weight.requires_grad = True
        else:
            self.word_lut = nn.Embedding(dicts.size(),
                                         opt.word_vec_size,
                                         padding_idx=s2s.Constants.PAD)
        self.feat_lut = nn.Embedding(64, 16, padding_idx=s2s.Constants.PAD)  # TODO: Fix this magic number
        input_size = input_size + 16 * 3
        self.rnn = nn.GRU(input_size, self.hidden_size,
                          num_layers=opt.layers,
                          dropout=opt.dropout,
                          bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, feats, hidden=None):
        """
        input: (wrap(srcBatch), wrap(srcBioBatch), lengths)
        """
        lengths = input[-1].data.view(-1).tolist()  # lengths data is wrapped inside a Variable
        wordEmb = self.word_lut(input[0])
        featsEmb = [self.feat_lut(feat) for feat in feats[0]]
        featsEmb = torch.cat(featsEmb, dim=-1)
        input_emb = torch.cat((wordEmb, featsEmb), dim=-1)
        emb = pack(input_emb, lengths)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        return hidden_t, outputs


class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1


class Decoder(nn.Module):
    def __init__(self, opt, dicts, weight):
        self.opt = opt
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.enc_rnn_size

        super(Decoder, self).__init__()
        if not (weight is None):
            self.word_lut = nn.Embedding.from_pretrained(weight)
            self.word_lut.weight.requires_grad = True
        else:
            self.word_lut = nn.Embedding(dicts.size(),
                                         opt.word_vec_size,
                                         padding_idx=s2s.Constants.PAD)
        self.rnn = StackedGRU(opt.layers, input_size, opt.dec_rnn_size, opt.dropout)
        self.attn = s2s.modules.ConcatAttention(opt.enc_rnn_size, opt.dec_rnn_size, opt.att_vec_size)
        self.attention = s2s.modules.MultiHeadAttention(opt.dec_rnn_size, opt.num_heads)
        self.dim_per_head = opt.dec_rnn_size * 6 // opt.num_heads
        self.n_heads = opt.num_heads
        self.W_V = nn.Linear(opt.dec_rnn_size, self.dim_per_head * self.n_heads)
        self.W_K = nn.Linear(opt.dec_rnn_size, self.dim_per_head * self.n_heads)

        self.dropout = nn.Dropout(opt.dropout)
        self.trans = nn.Linear((opt.enc_rnn_size + opt.dec_rnn_size), opt.dec_rnn_size)
        self.readout = nn.Linear((opt.enc_rnn_size + opt.dec_rnn_size + opt.word_vec_size), opt.dec_rnn_size)
        self.maxout = s2s.modules.MaxOut(opt.maxout_pool_size)
        self.maxout_pool_size = opt.maxout_pool_size

        self.tt = torch.cuda

        self.copySwitch = nn.Linear(opt.enc_rnn_size + opt.dec_rnn_size, 1)
        self.generator = nn.Sequential(
            nn.Linear(opt.dec_rnn_size // opt.maxout_pool_size, dicts.size()),  # TODO: fix here
            # nn.LogSoftmax(dim=1)
            nn.Softmax(dim=1)
        )
        self.hidden_size = opt.dec_rnn_size


    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, context, src_pad_mask, init_att, base_flag):

        emb_0 = self.word_lut(input[0][0].view(-1))
        tgt_len = input.size(0)

        g_outputs = []
        c_outputs = []
        mul_head_attns = []
        copyGateOutputs = []
        cur_context = init_att
        self.attention.applyMask(src_pad_mask)
        batch_size = init_att.size(0)
        k_s = self.W_K(context.transpose(0, 1)).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2)
        v_s = self.W_V(context.transpose(0, 1)).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2)

        mul_cs, mul_as = [],[]
        sample_y = []
        is_Copys = []
        all_pos = []
        # select_head = torch.randint(0, self.n_heads - 1, size=(batch_size,1,1))
        for i in range(tgt_len):
            if i==0:
                emb_t = emb_0
            else:
                emb_t = self.word_lut(sample_y[-1].view(-1))
            input_emb = emb_t
            if self.input_feed:

                input_emb = torch.cat([emb_t, cur_context], 1)

            output, hidden = self.rnn(input_emb, hidden)
            cur_context, context_attention, all_head_attn, mul_c, mul_a = self.attention(output.unsqueeze(1),k_s, v_s, base_flag)
            copyProb = self.copySwitch(torch.cat((output, cur_context), dim=1))
            copyProb = F.sigmoid(copyProb)

            readout = self.readout(torch.cat((emb_t, output, cur_context), dim=1))
            maxout = self.maxout(readout)
            output = self.dropout(maxout)

            g_prob = self.generator.forward(output)
            wordLk = torch.log(g_prob * ((1 - copyProb).expand_as(g_prob))+ 1e-8)
            copyLk = torch.log(context_attention * (copyProb.expand_as(context_attention))+ 1e-8)

            g_outputs += [wordLk]
            c_outputs += [copyLk]
            mul_head_attns += [all_head_attn]
            copyGateOutputs += [copyProb]
            mul_cs += [mul_c]
            mul_as += [mul_a]

            numWords = wordLk.size(1)
            numSrc = copyLk.size(1)
            numAll = numWords + numSrc
            allScores = torch.cat((wordLk, copyLk), dim=1)
            bestScores, bestScoresId = allScores.topk(1, 1, True, True)
            # bestScoresId is flattened beam x word array, so calculate which
            # word and beam each score came from
            prevK = bestScoresId / numAll
            # predict = bestScoresId - prevK * numWords
            predict = bestScoresId - prevK * numAll

            isCopy = predict.squeeze(1).ge(self.tt.LongTensor(wordLk.size(0)).fill_(numWords)).long()
            final_predict = predict.squeeze(1) * (1 - isCopy) + isCopy * s2s.Constants.UNK
            is_Copys += [isCopy]
            all_pos += [predict]
            sample_y.append(final_predict.view(-1,1))
        g_outputs = torch.stack(g_outputs)
        c_outputs = torch.stack(c_outputs)
        copyGateOutputs = torch.stack(copyGateOutputs)
        sample_y = torch.stack(sample_y)
        mul_cs = torch.stack(mul_cs)
        mul_as = torch.stack(mul_as)
        is_Copys = torch.stack(is_Copys)
        all_pos = torch.stack(all_pos)

        return sample_y, g_outputs, c_outputs, copyGateOutputs, hidden, context_attention, cur_context, mul_head_attns,is_Copys, all_pos, mul_cs, mul_as


class DecInit(nn.Module):
    def __init__(self, opt):
        super(DecInit, self).__init__()
        self.num_directions = 2 if opt.brnn else 1
        assert opt.enc_rnn_size % self.num_directions == 0
        self.enc_rnn_size = opt.enc_rnn_size
        self.dec_rnn_size = opt.dec_rnn_size
        self.initer = nn.Linear(self.enc_rnn_size // self.num_directions, self.dec_rnn_size)
        self.tanh = nn.Tanh()

    def forward(self, last_enc_h):
        # batchSize = last_enc_h.size(0)
        # dim = last_enc_h.size(1)
        return self.tanh(self.initer(last_enc_h))


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder, decIniter):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decIniter = decIniter

    def make_init_att(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.encoder.hidden_size * self.encoder.num_directions)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def forward(self, input):
        # ipdb.set_trace()
        src = input[0]
        tgt = input[2][0][:-1]  # exclude last target from inputs
        src_pad_mask = Variable(src[0].data.eq(s2s.Constants.PAD).transpose(0, 1).float(), requires_grad=False,
                                volatile=False)
        feats = input[1]
        enc_hidden, context = self.encoder(src, feats)

        init_att = self.make_init_att(context)
        enc_hidden = self.decIniter(enc_hidden[1]).unsqueeze(0)  # [1] is the last backward hiden

        sample_y, g_out, c_out, c_gate_out, dec_hidden, _attn, _attention_vector, mul_head_attns, isCopys, predCopyPositions,mul_cs,mul_as  = self.decoder(tgt, enc_hidden, context,
                                                                                      src_pad_mask, init_att,False)
        base_y, _, base_c_out, _, _, _, _, _,base_isCopys, base_predCopyPositions, _, _ = self.decoder(tgt, enc_hidden, context,src_pad_mask, init_att,True)
        return sample_y,isCopys, predCopyPositions, base_y, base_isCopys, base_predCopyPositions,g_out, c_out, base_c_out,c_gate_out, mul_head_attns,mul_cs, mul_as
