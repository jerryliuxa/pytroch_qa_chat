import torch
from data_preparation import build_vocabulary, text_to_indices, pad_sequences, simple_tokenizer
from model import TransformerModel, generate_square_subsequent_mask

# 加载模型
def load_model(vocab_size, embed_dim, nhead, num_layers, model_path='chatbot_model.pth', weights_only=True):
    model = TransformerModel(vocab_size, embed_dim, nhead, num_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 预测函数
def predict(model, input_text, vocabulary, tokenizer, max_len, device='cpu'):
    model.to(device)
    model.eval()
    input_indices = text_to_indices([input_text], vocabulary)
    padded_input_indices = pad_sequences(input_indices, max_len).to(device)
    
    # 首先通过嵌入层转换输入
    embedded = model.embedding(padded_input_indices)  # 这会得到正确的维度和数据类型
    embedded = model.pos_encoder(embedded)
    
    memory = model.encoder(embedded)
    
    sos_index = vocabulary['你好']  # 使用 "你好" 作为起始符
    eos_index = vocabulary['<PAD>']  # 使用 "<PAD>" 作为结束符
    
    tgt_tokens = torch.tensor([[sos_index]]).to(device)

    for _ in range(max_len):
        tgt_mask = generate_square_subsequent_mask(tgt_tokens.size(1)).to(device)
        
        # 对目标序列进行嵌入和位置编码
        tgt_embedded = model.embedding(tgt_tokens)
        tgt_embedded = model.pos_encoder(tgt_embedded)
        
        # 使用解码器
        output = model.decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
        
        # 使用全连接层生成下一个词的概率
        prob = model.fc(output[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        tgt_tokens = torch.cat([tgt_tokens, torch.tensor([[next_word]]).to(device)], dim=1)
        if next_word == eos_index:
            break
            
    
    
    index_to_word = {i: token for token, i in vocabulary.items()}
    predicted_text = ' '.join([index_to_word[token.item()] for token in tgt_tokens[0] if token.item() != eos_index])
    return predicted_text

if __name__ == '__main__':
    # 示例对话数据（用于构建词汇表）
    conversations = [
        ["你好", "你好啊"],
        ["今天天气不错", "是啊，很适合出去玩"],
        ["你叫什么名字", "我是机器人"],
        ["你会做什么", "我可以和你聊天"]
    ]
    inputs = [item[0] for item in conversations]
    outputs = [item[1] for item in conversations]

    # 构建词汇表
    vocabulary = build_vocabulary(inputs + outputs)
    vocab_size = len(vocabulary)
    max_len = max(max(len(simple_tokenizer(text)) if text else 0 for text in inputs), max(len(simple_tokenizer(text)) if text else 0 for text in outputs))

    # 加载模型
    embed_dim = 256
    nhead = 8
    num_layers = 2
    from data_preparation import SimpleTokenizer
    model = load_model(vocab_size, embed_dim, nhead, num_layers)
    tokenizer = SimpleTokenizer()

    # 进行预测
    input_text = "今天天气如何"
    predicted_text = predict(model, input_text, vocabulary, tokenizer, max_len, device='cpu')
    print(f"输入: {input_text}")
    print(f"预测输出: {predicted_text}")
