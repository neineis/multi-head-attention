import torch
import torch.nn as nn
import math
import numpy as np

class ScaledDotProductAttention(nn.Module):  # 点乘
    def __init__(self,d_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.d_dim = d_dim
        self.weight_linear = nn.Linear(self.d_dim,1)
        self.sm = nn.Softmax(dim=1)

    def computeAttWeight(self,Q):
        weights = self.weight_linear(Q)
        weights = self.sm(weights)
        return weights

    def forward(self, Q, K, V, scale =None,attn_mask=None):  # 实现注意力公式
        """前向传播.

        Args:
            q: Queries张量，形状为[B, L_q, h_num, D_q]
            k: Keys张量，形状为[B, L_k, h_num, D_k]
            v: Values张量，形状为[B, L_v, h_num, D_v]，一般来说就是k
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

        if scale:
            scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
        if attn_mask:
            scores.masked_fill_(attn_mask, -np.inf)
        attn = nn.Softmax(dim=-1)(scores)
        # weight_attn = torch.mul(attn, weights)

        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):  # 多头注意力
    def __init__(self, d_model, n_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model% n_heads == 0
        self.dim_per_head = d_model // n_heads
        self.dropout = dropout
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, self.dim_per_head * self.n_heads)
        self.W_K = nn.Linear(d_model, self.dim_per_head * self.n_heads)
        self.W_V = nn.Linear(d_model, self.dim_per_head * self.n_heads)
        self.dot_product_attention = ScaledDotProductAttention(self.dim_per_head)

        self.linear_final1 = nn.Linear(self.dim_per_head, self.dim_per_head)
        self.linear_final2 = nn.Linear(self.dim_per_head, d_model)
        self.linear_weight = nn.Linear(self.dim_per_head,1)

        self.dropout = nn.Dropout(self.dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, Q, K, V, attn_mask=None):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2)
        if attn_mask:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        scale = q_s.size(-1)** -0.5
        context, attn= self.dot_product_attention(q_s, k_s, v_s, scale,attn_mask)
        context = context.sum(2)
        head_weights =  self.linear_weight(context)
        head_weights =  nn.Softmax(dim=1)(head_weights)
        context = torch.mul(context, head_weights)
        context = context.sum(1)
        attn = attn.squeeze(2)
        '''
        attn: [B, h_num, L_v]
        context :[B, D_v]
        ret_attn: type(list) ,length = n_heads, n_heads * B * L_v
        '''
        # context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.dim_per_head)
        output = self.linear_final1(context)
        output = self.linear_final2(output)
        output = self.dropout(output)
        ret_attn = torch.mul(attn, head_weights)
        ret_attn = ret_attn.sum(1)
        attn = torch.split(attn, 1, 1)
        attn = [x.squeeze(1) for x in attn]
        # output = self.layer_norm(residual + output)
        return output,  ret_attn, attn