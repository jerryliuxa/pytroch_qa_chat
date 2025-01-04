import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import data_preparation
from data_preparation import prepare_data, build_vocabulary, text_to_indices, pad_sequences, ChatbotDataset, simple_tokenizer
from model import TransformerModel, generate_square_subsequent_mask

# 在调用 pad_sequences 之前定义 max_len
max_len = 128  # 假设我们选择 128 作为最大长度

# 超参数
batch_size = 32
learning_rate = 1e-4
epochs = 10
embed_dim = 256
nhead = 8
num_layers = 2  # 直接使用整数
dropout = 0.1

# 从 open_qa.jsonl 文件加载数据
file_path = 'data/HC3-Chinese/open_qa.jsonl'
inputs, outputs = prepare_data(file_path)

# 构建词汇表
vocabulary = build_vocabulary(inputs + outputs)
vocab_size = len(vocabulary)

# 将文本转换为索引并填充
input_indices = text_to_indices(inputs, vocabulary)
output_indices = text_to_indices(outputs, vocabulary)
max_len = max(len(input) for input in input_indices)
padded_input_indices = pad_sequences(input_indices, max_len)
padded_output_indices = pad_sequences(output_indices, max_len)

# 创建Dataset和DataLoader
tokenizer = simple_tokenizer
global_max_len = max(max(len(seq) for seq in input_indices), max(len(seq) for seq in output_indices))
dataset = ChatbotDataset(inputs, outputs, tokenizer, global_max_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
model = TransformerModel(vocab_size, embed_dim, nhead, num_layers, dropout)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding的loss
optimizer = AdamW(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        # 生成mask
        src_padding_mask = (src == 0)
        tgt_padding_mask = (tgt_input == 0)
        look_ahead_mask = generate_square_subsequent_mask(tgt_input.size(1))

        # 运行模型
        print("src shape:", src.shape)
        print("tgt_input shape:", tgt_input.shape)
        print("tgt_expected shape:", tgt_expected.shape)
        print("look_ahead_mask shape:", look_ahead_mask.shape)
        print("src_padding_mask shape:", src_padding_mask.shape)
        print("tgt_padding_mask shape:", tgt_padding_mask.shape)
        output = model(src, tgt_input, tgt_mask=look_ahead_mask,
                       src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)

        # 计算loss
        loss = criterion(output.reshape(-1, vocab_size), tgt_expected.reshape(-1))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch: {epoch+1}, Loss: {total_loss/len(dataloader)}")

    # 保存模型
    torch.save(model.state_dict(), 'chatbot_model.pth')
