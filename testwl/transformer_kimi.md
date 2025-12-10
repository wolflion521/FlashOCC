我会像搭积木一样，手把手带你从零写出Transformer。咱们把这张架构图**拆解成7个可运行的代码块**，每个块对应图中的一个组件。最后有**三种写法**供你选择，从"纯手工"到"直接调用库"。

---

## **第一步：必备基础（5分钟）**

先确保你理解这三个PyTorch操作，代码里会高频出现：

```python
import torch
import torch.nn as nn
import math

# 1. 矩阵乘法
# Q @ K^T  :  (batch, seq_len, d_model) @ (batch, d_model, seq_len) → (batch, seq_len, seq_len)
Q = torch.randn(2, 10, 512)  # 2个句子，每句10个词，512维
K = torch.randn(2, 10, 512)
scores = torch.matmul(Q, K.transpose(-2, -1))  # 这就是注意力分数！

# 2. 缩放
scores = scores / math.sqrt(512)  # 防止梯度爆炸

# 3. Softmax
attention = torch.softmax(scores, dim=-1)  # 每行和为1，表示权重
```

---

## **第二步：对照架构图，逐个实现7个组件**

### **组件1：Embedding层（图的左下角"Input Embedding"）**
作用：把单词id变成512维向量

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        # vocab_size: 词典大小（比如中英混合模型可能有50000）
        # d_model: 向量维度（Transformer默认512）
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        # x: (batch, seq_len) 单词id
        return self.embedding(x) * math.sqrt(self.d_model)
        # 乘以sqrt(d_model)是论文里的小技巧，防止embedding值太小
```

### **组件2：位置编码（图的"Positional Encoding"）**
作用：给词向量加位置信息（因为attention本身不知道词序）

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        super().__init__()
        # 创建位置矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # 用sin/cos函数生成不同频率的波动
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维cos
        
        self.register_buffer('pe', pe.unsqueeze(0))  # 注册为buffer，不参与训练
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]  # 逐元素相加
```

### **组件3：缩放点积注意力（Scaled Dot-Product Attention）**
对应图中**Multi-Head Attention**里的核心计算

```python
def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    """
    Q, K, V: (batch, seq_len, d_model)
    mask: (batch, 1, seq_len) 或 (batch, seq_len, seq_len)，用于遮挡未来词
    """
    d_k = Q.size(-1)  # 通常是 d_model / nhead
    
    # 1. 计算注意力分数 (batch, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. 应用mask（decoder用，encoder不用）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # 把要遮挡的位置设为负无穷
    
    # 3. Softmax得到权重
    attention = torch.softmax(scores, dim=-1)
    
    # 4. 应用dropout防止过拟合
    if dropout is not None:
        attention = dropout(attention)
    
    # 5. 加权求和 (batch, seq_len, d_model)
    output = torch.matmul(attention, V)
    
    return output, attention  # 返回 attention 权重用于可视化
```

### **组件4：多头注意力（Multi-Head Attention）**
图中**Multi-Head Attention**的完整实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0  # 必须能整除
        
        self.d_k = d_model // nhead  # 每个头的维度，64
        self.nhead = nhead
        
        # 4个线性层：Q、K、V各一个，最后output一个
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        query, key, value: (batch, seq_len, d_model)
        mask: (batch, 1, seq_len) 或 None
        """
        batch_size = query.size(0)
        
        # 1. 线性变换得到Q, K, V
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. 拆成多头 (batch, nhead, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # 3. 对每个头独立做注意力
        attention_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask, self.dropout
        )
        
        # 4. 合并多头 (batch, seq_len, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.nhead * self.d_k
        )
        
        # 5. 最后线性变换
        output = self.W_o(attention_output)
        
        return output, attention_weights
```

### **组件5：前馈网络（Feed Forward）**
图中**Feed Forward**模块

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        # d_ff是中间层维度，通常是d_model的4倍
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # (batch, seq_len, d_model) → (batch, seq_len, d_ff) → (batch, seq_len, d_model)
        return self.w2(self.dropout(torch.relu(self.w1(x))))
```

### **组件6：Add & Norm层（残差连接+层归一化）**
图中每个模块后面的**Add & Norm**

```python
class AddNorm(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)  # 层归一化
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        # sublayer是一个函数（比如MultiHeadAttention或FeedForward）
        # 残差连接：先过子层，再dropout，再加回原输入，最后norm
        return self.norm(x + self.dropout(sublayer(x)))
```

### **组件7：Mask生成（Decoder用）**
防止decoder偷看未来词

```python
def create_look_ahead_mask(seq_len):
    """
    创建一个上三角矩阵，对角线以上全是0（要遮挡）
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    # diagonal=1表示保留对角线，对角线以上设为0（遮挡）
    return mask == 0  # 返回True/False矩阵，True表示允许注意
```

---

## **第三步：组装成Encoder和Decoder**

### **Encoder层（图的左半部分Nx个方块）**
```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        
    def forward(self, x, mask=None):
        # 1. 自注意力 + Add & Norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.add_norm1(x, lambda x: attn_output)
        
        # 2. 前馈网络 + Add & Norm
        x = self.add_norm2(x, self.feed_forward)
        
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers=6, d_model=512, nhead=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
```

### **Decoder层（图的右半部分Nx个方块）**
```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, d_ff=2048, dropout=0.1):
        super().__init__()
        # 1. Masked Self-Attention
        self.masked_self_attention = MultiHeadAttention(d_model, nhead, dropout)
        # 2. Encoder-Decoder Attention
        self.enc_dec_attention = MultiHeadAttention(d_model, nhead, dropout)
        # 3. Feed Forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)
        
    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        # 1. Masked Self-Attention + Add & Norm
        attn1, _ = self.masked_self_attention(x, x, x, look_ahead_mask)
        x = self.add_norm1(x, lambda x: attn1)
        
        # 2. Encoder-Decoder Attention + Add & Norm
        # Q来自decoder，K,V来自encoder
        attn2, _ = self.enc_dec_attention(x, enc_output, enc_output, padding_mask)
        x = self.add_norm2(x, lambda x: attn2)
        
        # 3. Feed Forward + Add & Norm
        x = self.add_norm3(x, self.feed_forward)
        
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers=6, d_model=512, nhead=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)
        ])
        
    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, look_ahead_mask, padding_mask)
        return x
```

---

## **第四步：三种完整Transformer写法对比**

### **写法1：纯手工（最推荐学习）**
```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.encoder = Encoder(num_encoder_layers, d_model, nhead, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, nhead, d_ff, dropout)
        
        self.final_linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 1. Encoder
        src_emb = self.dropout(self.pos_encoding(self.token_embedding(src)))
        enc_output = self.encoder(src_emb, src_mask)
        
        # 2. Decoder
        tgt_emb = self.dropout(self.pos_encoding(self.token_embedding(tgt)))
        dec_output = self.decoder(tgt_emb, enc_output, tgt_mask, src_mask)
        
        # 3. Final Linear + Softmax
        output = self.final_linear(dec_output)
        return torch.softmax(output, dim=-1)
```

### **写法2：用PyTorch内置nn.Transformer（生产推荐）**
```python
# PyTorch 1.2+ 内置了高效实现
class TransformerTorch(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 直接调用官方实现（CUDA优化过，速度快）
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True  # PyTorch 1.9+支持，输入格式为(batch, seq, feature)
        )
        
        self.final_linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.pos_encoding(self.embedding(src))
        tgt_emb = self.pos_encoding(self.embedding(tgt))
        
        # 官方接口：src, tgt, src_mask, tgt_mask, memory_mask...
        output = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask)
        
        return torch.softmax(self.final_linear(output), dim=-1)
```

**参数对比**：
```python
# nn.Transformer参数
nn.Transformer(
    d_model=512,        # 向量维度
    nhead=8,            # 头数
    num_encoder_layers=6,  # encoder层数
    num_decoder_layers=6,  # decoder层数
    dim_feedforward=2048,  # 前馈网络维度
    dropout=0.1,        # dropout率
    activation='relu',  # 激活函数
    batch_first=True    # 输入格式 (batch, seq, feature)
)
```

### **写法3：Hugging Face（最省事）**
```python
from transformers import BertModel

# 一行代码加载预训练模型
model = BertModel.from_pretrained('bert-base-uncased')

# 但黑盒，不利于学习Transformer原理
```

---

## **第五步：参数详解与调参建议**

| 参数 | 含义 | 典型值 | 调参技巧 |
|------|------|--------|----------|
| **d_model** | 向量维度 | 512 | 越大越慢，768/1024用于大模型 |
| **nhead** | 头数 | 8 | 必须整除d_model，head太多会退化 |
| **num_layers** | 层数 | 6 | 深层(12)效果好但难训练，残差要稳 |
| **d_ff** | 前馈维度 | 2048 | 通常d_model×4 |
| **dropout** | 正则化 | 0.1 | 深层用0.15-0.2，防止过拟合 |
| **max_len** | 位置编码长度 | 5000 | 按数据集最大长度设，太长浪费 |

---

## **第六步：训练循环示例**

```python
# 超参数
vocab_size = 5000
d_model = 512
nhead = 8
num_layers = 6

# 实例化模型
model = Transformer(vocab_size, d_model, nhead, num_layers)

# 损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# 模拟数据
src = torch.randint(0, vocab_size, (32, 10))  # 32个样本，每句10词
tgt = torch.randint(0, vocab_size, (32, 10))

# 生成mask
def create_masks(src, tgt):
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # pad token=0
    tgt_mask = create_look_ahead_mask(tgt.size(1)).to(tgt.device)
    return src_mask, tgt_mask

# 训练循环
for epoch in range(100):
    optimizer.zero_grad()
    
    src_mask, tgt_mask = create_masks(src, tgt)
    output = model(src, tgt[:, :-1], src_mask, tgt_mask)  # tgt去尾，防止信息泄露
    
    loss = criterion(output.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## **总结：三种写法的取舍**

| 写法 | 适合场景 | 优点 | 缺点 |
|------|----------|------|------|
| **纯手工** | **学习、发论文改造** | 完全可控，理解透彻 | 慢、易错 |
| **nn.Transformer** | **工业落地** | 官方优化快、稳定 | 黑盒，不易改 |
| **Hugging Face** | **快速Demo** | 一行代码、预训练多 | 最黑盒，调不动 |

 **建议**  ：
1. **先抄纯手工代码3遍**，默写 `scaled_dot_product_attention` 和 `MultiHeadAttention`
2. **再换成nn.Transformer**，对比输出是否一致（验证理解）
3. **面试时**，能手写前两种，说清参数含义，就**稳过90%公司**

Transformer代码难，是因为**细节多但规律强**。记住核心：**embedding + positional → multi-head attention → feed forward → add & norm**，循环6次。每次写代码，边看图边对号入座，三次就熟练。