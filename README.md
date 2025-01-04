# 创建 README.md 文件

这是一个基于Transformer的聊天机器人模型，使用PyTorch框架实现。该项目包括数据准备、模型训练和预测功能。

## 目录

- [功能](#功能)
- [环境要求](#环境要求)
- [使用方法](#使用方法)
- [代码结构](#代码结构)
- [贡献](#贡献)
- [许可证](#许可证)

## 功能

- 数据准备：从 `data/HC3-Chinese/open_qa.jsonl` 文件中加载对话数据，构建词汇表并将文本转换为索引。
- 模型训练：使用Transformer模型进行训练。
- 预测：根据输入文本生成响应。

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- transformers
- jieba（用于中文分词）

## 使用方法

1. **克隆项目**

   ```bash
   git clone <项目地址>
   cd <项目目录>
   ```

2. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

3. **准备数据**

   运行 `data_preparation.py` 文件，该文件会从 `data/HC3-Chinese/open_qa.jsonl` 加载数据，构建词汇表并将文本转换为索引序列。

4. **训练模型**

   运行 `train.py` 以训练模型。训练完成后，模型将保存在 `chatbot_model.pth` 中。当前使用的超参数包括：

   - 批量大小 (`batch_size`): 32
   - 学习率 (`learning_rate`): 1e-4
   - 训练轮数 (`epochs`): 10
   - 嵌入维度 (`embed_dim`): 256
   - 注意力头数 (`nhead`): 8
   - Transformer层数 (`num_layers`): 2
   - Dropout率 (`dropout`): 0.1

   ```bash
   python train.py
   ```

5. **进行预测**

   运行`predict.py`以使用训练好的模型进行预测。

   ```bash
   python predict.py
   ```

## 代码结构

- `train.py`：模型训练脚本。
- `predict.py`：模型预测脚本。
- `data_preparation.py`：数据准备和处理功能。
- `model.py`：定义Transformer模型及其相关功能。
- `data/HC3-Chinese/open_qa.jsonl`：用于模型训练的开放领域问答数据集。
- `.gitignore`：Git忽略文件。

## 贡献

欢迎任何形式的贡献！请提交问题或拉取请求。

## 许可证

本项目采用MIT许可证，详细信息请查看LICENSE文件。
