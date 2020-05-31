import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

class ConcatAttention(nn.Module):  # 点乘
    def __init__(self,d_dim):
        super(ConcatAttention, self).__init__()
        self.d_dim = d_dim
        self.linear_pre = nn.Linear(d_dim, d_dim, bias=True)
        self.linear_q = nn.Linear(d_dim, d_dim, bias=False)
        self.linear_v = nn.Linear(d_dim, 1, bias=False)
        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()


    def forward(self, Q, K, V, scale, mask, precompute=None):  # 实现注意力公式
        """前向传播.

        Args:
            q: Queries张量，形状为[B, h_num, L_q, D_q]
            k: Keys张量，形状为[B,  h_num, L_k,D_k]
            v: Values张量，形状为[B, h_num,  L_v,D_v]，一般来说就是k
            scale: 缩放因子，一个浮点标量
            attn_mask: Masking张量，形状为[B, L_q, h_num, L_k]

        Returns:
            上下文张量和attetention张量
            attn: scores张量, 形状为[B, h_num, L_q, L_v]
            context: context, 形状为[B, h_num, L_q, D_v]
        """
        if precompute is None:
            precompute00 = self.linear_pre(K)
            precompute = precompute00.view(K.size(0), K.size(1),  -1, self.d_dim)  # batch x h_num * sourceL x att_dim

        targetT = self.linear_q(Q)  # batch x h_num * 1 x att_dim

        tmp10 = precompute + targetT.expand_as(precompute)  # batch x h_num x sourceL x att_dim
        tmp20 = self.tanh(tmp10)  # batch x h_num x sourceL x att_dim
        energy = self.linear_v(tmp20).view(tmp20.size(0), tmp20.size(1),-1,tmp20.size(2))  # batch x h_num x sourceL
        if mask is not None:
            energy = energy * (1 - mask) + mask * (-1000000)

        score = self.sm(energy)
        score_m = score.view(score.size(0),score.size(1), 1, score.size(3))  # batch x h_num x 1 x sourceL

        weightedContext = torch.matmul(score_m, V)    # batch x h_num x 1 x dim

        return weightedContext, score_m, precompute



class ScaledDotProductAttention(nn.Module):  # 点乘
    def __init__(self,d_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.d_dim = d_dim
        self.weight_linear = nn.Linear(self.d_dim,1)
        self.sm = nn.Softmax(dim=1)
        self.mask = None

    def forward(self, Q, K, V, scale, mask):  # 实现注意力公式
        """前向传播.

        Args:
            q: Queries张量，形状为[B, h_num, L_q, D_q]
            k: Keys张量，形状为[B, h_num, L_k,  D_k]
            v: Values张量，形状为[B, h_num, L_v, D_v]，一般来说就是k
            scale: 缩放因子，一个浮点标量
            attn_mask: Masking张量，形状为[B, L_q, h_num, L_k]

        Returns:
            上下文张量和attetention张量
            attn: scores张量, 形状为[B, h_num, L_q, L_v]
            context: context, 形状为[B, h_num, L_q, D_v]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2))
        # weights = self.computeAttWeight(Q)
        # weights = weights.expand(scores.shape)
        # self.mask: batch_size, seq_len
        if scale:
            scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
        if mask is not None:
            scores = scores * (1 - mask) + mask * (-1000000)

        attn = nn.Softmax(dim=-1)(scores)
        # weight_attn = torch.mul(attn, weights)

        context = torch.matmul(attn, V)

        return context, attn


class MultiHeadAttention(nn.Module):  # 多头注意力
    def __init__(self, d_model, n_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model* 6 % n_heads == 0
        self.dim_per_head = d_model * 6 // n_heads
        self.dropout = dropout
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, self.dim_per_head * self.n_heads )
        self.dot_product_attention = ScaledDotProductAttention(self.dim_per_head)

        self.linear_final1 = nn.Linear(self.dim_per_head, self.dim_per_head)
        self.linear_final2 = nn.Linear(self.dim_per_head, d_model)
        self.linear_weight = nn.Linear(self.dim_per_head,1)

        self.dropout = nn.Dropout(self.dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(d_model)

    def applyMask(self, mask):
        self.mask = mask
        self.mask = self.mask.unsqueeze(1).unsqueeze(1).repeat(1, self.n_heads, 1, 1)



    def forward(self, Q, k_s, v_s, base_flag):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2)

        scale = q_s.size(-1)** -0.5

        context, attn = self.dot_product_attention(q_s, k_s, v_s, scale, self.mask)
        context = context.squeeze(2)   # batch_size, num_heads, dim_pre_head
        weights = self.linear_weight(context).squeeze(2)
        if not base_flag:
            select_head = torch.multinomial(F.softmax(weights), 1).unsqueeze(1)
        else:
            _, cur_max_y = torch.max(F.softmax(weights), 1)
            select_head = cur_max_y.view(-1,1).unsqueeze(1)
        select_head1 = select_head.repeat(1, 1, self.dim_per_head).cuda()
        output = torch.gather(context, dim=1, index=select_head1).squeeze(1)
        output = self.linear_final1(output)
        output = self.linear_final2(output)
        output = self.dropout(output)

        attn = attn.squeeze(2)
        mul_attn = attn
        select_head2 = select_head.repeat(1, 1, attn.size(2)).cuda()
        ret_attn = torch.gather(attn, dim=1, index=select_head2).squeeze(1)


        attn = torch.split(attn, 1, 1)
        attn = [x.squeeze(1) for x in attn]


        '''
        attn: [B, h_num, L_v]
        context :[B, D_v]
        ret_attn: type(list) ,length = n_heads, n_heads * B * L_v
        '''
        # context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.dim_per_head)
        # output = self.layer_norm(residual + output)
        return output,  ret_attn, attn, context, mul_attn