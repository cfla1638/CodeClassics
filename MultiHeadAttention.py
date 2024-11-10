# Multi-Head Attention
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_head: int, dropout = 0.0) -> None:
        """
        Parameters:
         - hidden_dim: 输入特征的维度
         - num_head : 头的数目
         - dropout
        """
        super().__init__()

        self.hidden_dim = hidden_dim    # 输入序列特征的维度
        self.num_head = num_head        # 头数
        self.head_dim = hidden_dim // num_head      # 一个头处理的特征维度数

        self.q_embd_net = nn.Linear(hidden_dim, hidden_dim)
        self.k_embd_net = nn.Linear(hidden_dim, hidden_dim)
        self.v_embd_net = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.o_prog = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input: torch.Tensor, mask: torch.Tensor = None):
        """
        Parameters:
         - input: (batch_size, seq_len, hidden_dim)
         - mask: (batch_size, seq_len)
        """
        batch_size, seq_len, _ = input.shape

        # Q, K, V = (batch_size, seq_len, hidden_dim)
        Q = self.q_embd_net(input)
        K = self.k_embd_net(input)
        V = self.v_embd_net(input)

        # Q, K, V = (batch_size, num_head, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)

        # attention_weight = (batch_size, num_head, seq_len, seq_len)
        attention_weight = (Q @ K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # 处理掩码
        if mask is not None:
            attention_weight.masked_fill(
                mask.unsqueeze(1).unsqueeze(2) == 0, # 扩展成(batch_size, 1, 1, seq_len)进行广播
                float("-inf")
            )

        # attention_weight = (batch_size, num_head, seq_len, seq_len)
        attention_weight = torch.softmax(attention_weight, dim = -1)
        attention_weight = self.dropout(attention_weight)
        
        # output: (batch_size, num_head, seq_len, head_dim)
        output = attention_weight @ V
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.o_prog(output)


if __name__ == '__main__':
    mask = torch.zeros(256, 7)
    mask[:, 0:3] = 1
    t = torch.randn(256, 7, 32)
    net = MultiHeadAttention(32, 8)
    print(net(t, mask).shape)
