import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers: int, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True)
        decoder_layer = TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True)
        
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, 
                src_padding_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        src_embedded = self.dropout(self.pos_encoder(self.embedding(src)))
        tgt_embedded = self.dropout(self.pos_encoder(self.embedding(tgt)))
        
        memory = self.encoder(src_embedded, src_key_padding_mask=src_padding_mask)
        output = self.decoder(tgt_embedded, memory, 
                            tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_padding_mask,
                            memory_key_padding_mask=memory_padding_mask)
        
        return self.fc(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

if __name__ == '__main__':
    vocab_size = 100  # 示例词汇表大小
    embed_dim = 128
    nhead = 8
    num_layers = 2
    model = TransformerModel(vocab_size, embed_dim, nhead, num_layers)
    print(model)

    # 示例输入
    src = torch.randint(0, vocab_size, (2, 10))  # 批次大小为2，序列长度为10
    tgt = torch.randint(0, vocab_size, (2, 10))
    output = model(src, tgt)
    print("模型输出形状:", output.shape)

    mask = generate_square_subsequent_mask(tgt.size(1))
    print("掩码形状:", mask.shape)
    print("示例掩码:", mask)
