import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import jieba
import re

class ChatbotDataset(Dataset):
    def __init__(self, inputs, labels, tokenizer, max_len):
        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocabulary = build_vocabulary(inputs + labels)
        self.encoded_inputs = [text_to_indices([text], self.vocabulary)[0] for text in inputs]
        self.encoded_labels = [text_to_indices([text], self.vocabulary)[0] for text in labels]
        self.max_len = max_len
        self.encoded_inputs = [self._pad_sequence(input_ids) for input_ids in self.encoded_inputs]
        self.encoded_labels = [self._pad_sequence(label_ids) for label_ids in self.encoded_labels]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded_inputs[idx]), torch.tensor(self.encoded_labels[idx])

    def _pad_sequence(self, sequence):
        padding_length = self.max_len - len(sequence)
        return sequence + [0] * padding_length

def simple_tokenizer(text):
    return jieba.lcut(text)

def build_vocabulary(texts):
    token_counts = Counter()
    for text in texts:
        token_counts.update(simple_tokenizer(text))
    vocabulary = {token: i + 2 for i, token in enumerate(token_counts)}
    vocabulary['<PAD>'] = 0
    vocabulary['<UNK>'] = 1
    return vocabulary

def text_to_indices(texts, vocabulary):
    indices = []
    for text in texts:
        tokens = simple_tokenizer(text)
        indices.append([vocabulary.get(token, vocabulary['<UNK>']) for token in tokens])
    return indices

def pad_sequences(sequences, max_len):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            padded_sequences.append(seq + [0] * (max_len - len(seq)))
        else:
            padded_sequences.append(seq[:max_len])
    return torch.tensor(padded_sequences)

import json

def prepare_data(file_path):
    inputs = []
    outputs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                question = data['question']
                if data['human_answers'] and data['human_answers'][0]:
                    answer = data['human_answers'][0]
                    inputs.append(question)
                    outputs.append(answer)
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line}")
    return inputs, outputs

class SimpleTokenizer:
    def __call__(self, text):
        vocab = build_vocabulary([text])
        indices = text_to_indices([text], vocab)
        return indices[0]

if __name__ == '__main__':
    # 从 open_qa.jsonl 文件加载数据
    file_path = 'data/HC3-Chinese/open_qa.jsonl'
    inputs, outputs = prepare_data(file_path)

    # 构建词汇表
    vocabulary = build_vocabulary(inputs + outputs)
    print("词汇表大小:", len(vocabulary))

    # 将文本转换为索引
    input_indices = text_to_indices(inputs, vocabulary)
    output_indices = text_to_indices(outputs, vocabulary)

    # 填充序列
    max_len = max(max(len(seq) for seq in input_indices), max(len(seq) for seq in output_indices))
    padded_input_indices = pad_sequences(input_indices, max_len)
    padded_output_indices = pad_sequences(output_indices, max_len)

    print("填充后的输入索引:", padded_input_indices)
    print("填充后的输出索引:", padded_output_indices)

    tokenizer = SimpleTokenizer()
    dataset = ChatbotDataset(inputs, outputs, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        input_ids, labels = batch
        print("批次输入:", input_ids)
        print("批次标签:", labels)
        break
