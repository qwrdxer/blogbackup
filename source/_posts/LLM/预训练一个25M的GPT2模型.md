---
title: 预训练一个25M的GPT2模型
categories:
  - LLM
date: 2025-04-07 14:19:54
tags:
---



### 参考
https://www.modelscope.cn/datasets/gongjy/minimind_dataset/summary
https://github.com/jingyaogong/minimind?tab=readme-ov-file
### 数据准备

#### 数据集选择
使用pretrain_hq.jsonl 进行预训练
https://www.modelscope.cn/datasets/gongjy/minimind_dataset/summary
样本例子如下:都是中文文本,每一行一个JSON

```json
{"text": "<s>介绍一本你近期阅读的书籍。\n最近我阅读了一本名为《霹雳娇娃》的小说。这本小说讲述了三位女性侠士与神秘组织之间的斗争。在这个故事中，女性侠士们并不是柔弱无力的弱女子，她们展现出了超强的战斗能力和深厚的情感内涵。小说通过复杂的情节和令人窒息的危机场面，揭示了人性善恶的多重性，并强调了团结互助的重要性。\n这本小说的细节描写十分生动鲜活，使读者能够身临其境。此外，角色之间的互动也表现出了情感和人性的复杂性。我认为这本小说不仅仅是一个好的故事，更是一部能够启示读者思考的良心之作。\n基于以上这段文本换一种方式重新描述。最近，我读了一本非常引人入胜的小说——《霹雳娇娃》。这本小说跟随三位女性侠士与神秘组织之间的生死决战。故事展示了三位侠士的坚韧和无所畏惧的战斗精神，同时也揭示了隐藏的人性弱点和矛盾之处。情节丰满，紧张刺激——你会感到自己置身其中。同时，本书还强调了互助合作的重要性，切中当今社会和我们内心深处的共鸣。\n在精彩的情节中，作者巧妙地把握了角色之间的情感和复杂性，使得书中的事件和人物不再只是一个干瘪的故事线，而是变成了一个立体的故事世界。这本小说不仅可以带给你好的故事，还可以让读者了解到自己内心深处的一些意义。</s>"}
{"text": "<s>常见的心理障碍类型有哪些？\n心理障碍常常是人类生活压力过大、心理压抑等原因造成的。常见的心理障碍类型包括：抑郁症、强迫症、焦虑症、创伤后应激障碍等。其中，抑郁症主要表现为情感低落、失去兴趣等；强迫症则表现为强迫症状，需要反复进行某种行为；焦虑症则表现为极度担忧、不安和消极情绪；创伤后应激障碍则是由于遭受创伤事件后造成的精神创伤而引起的精神病症状。处理心理障碍需要采取适当的心理疏导和药物辅助治疗。如果你或身边的人需要帮助，及时与心理医生或心理咨询师联系是很重要的。\n基于以上这段文本重新润色，保证意思一致。常见的心理障碍类型有哪些？\n当人们的生活压力过大或情绪被压抑时，容易出现各种心理障碍。到目前为止，我们已经确定了几种常见的心理障碍类型，它们分别是抑郁症、强迫症、焦虑症和创伤后应激障碍。抑郁症表现为情感低落和失去兴趣；强迫症表现为需要反复进行某种行为；焦虑症表现为极度担忧、不安和消极情绪；创伤后应激障碍则是受到创伤事件影响后引起的精神创伤。对于这些心理障碍，心理医生和心理咨询师可以提供适当的心理疏导和药物辅助治疗。即使你周围的人并没有得到心理障碍的诊断，但如果你感觉他们表现出这些征兆，还是尽早提供帮助为好。</s>"}
```

编写一个简单的脚本看一下平均长度
```python
# Python 实现
def calculate_average_length_and_line_count(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    line_count = len(lines)
    line_lengths = [len(line.strip()) for line in lines]
    avg_length = sum(line_lengths) / line_count if line_count > 0 else 0
    
    return line_count, avg_length

# 输入文件路径
file_path = '/datasets/pretrain_hq.jsonl'
line_count, average_length = calculate_average_length_and_line_count(file_path)
print(f"总行数: {line_count}, 每一行的平均长度是: {average_length}")

```
总行数: 1,413,103, 每一行的平均长度是: 416.641455718373
#### 思路设计
- 我们的模型上下文设置为512 ,大于512的行需要截取，小于512的行需要填充(padding)

**首先是Dataset的设计**
- Dataset提供给模型的X,Y应该是Tokenizer后的向量对，因此需要一个Tokenizer
- 根据文本长度做padding或者截取操作

```python
import torch
from torch.utils.data import Dataset
import json
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from typing import List
class PretrainDataset(Dataset):
    def __init__(self,data_dir:str,max_length:int,tokenizer:PreTrainedTokenizerFast):
        super().__init__()
        self.data_dir=data_dir
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.samples=self.load_data(self.data_dir)
    def load_data(self,path:str)->List[str]:
        samples=[]
        with open(path,'r',encoding='utf-8') as f:
            for idx,line in enumerate(f,1):
                data=json.load(line.strip())
                samples.append(data['text'])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample=self.samples[index]
        sample_tokenize = self.tokenizer(
            sample,  
            max_length=self.max_length,  # 设置输入序列的最大长度，超过该长度的文本将会被截断，默认值为 None，表示不限制最大长度
            padding='max_length',  # 设置是否对输入进行填充以及填充方式。'max_length' 表示填充至 `max_length` 指定的长度；'longest' 会根据输入序列的最大长度进行填充，'do_not_pad' 表示不进行填充。
            truncation=True,  # 设置是否对输入进行截断。True 表示超出 `max_length` 的部分会被截断
            return_tensors='pt'  # 设置返回的格式。'pt' 表示返回 PyTorch 张量（Tensor）。可以选择 'tf'（返回 TensorFlow 张量）或 'np'（返回 NumPy 数组），'pt' 是针对 PyTorch 框架的
        )

        #{'input_ids': tensor([[2157, 1293,    0,    0,    0,    0,    0,    0,    0,    0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
        input_ids=sample_tokenize['input_ids'].squeeze()
        loss_mask=sample_tokenize['attention_mask'].squeeze()
        # X[n] 与Y[n+1] 一一对应 ,即训练的目的是next token预测
        # Y[b]与loss_mask[b]一一对应,padding部分的token不应该参与loss计算
        X=sample_tokenize[:-1]
        Y=sample_tokenize[1:]
        loss_mask=loss_mask[1:]
        return X,Y,loss_mask
```

### 模型参数设计
```python
batch_size = 64 # 每次训练的样本数
context_length=256# 上下文长度
d_model=512 # 嵌入向量的维度
num_block=8 # Transformer块数
num_heads=16 # 多头注意力的头数
max_token_value=6400
```

打印一下模型的参数量
```python
from minimodel import GPT2Model
import torch
batch_size = 64 # 每次训练的样本数
context_length=256# 上下文长度
d_model=512 # 嵌入向量的维度
num_block=8 # Transformer块数
num_heads=16 # 多头注意力的头数
max_token_value=6400
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Generate
model = GPT2Model(
    max_token_value=max_token_value,
    d_model=d_model,
    num_heads=num_heads,
    num_blocks=num_block,
    context_length=context_length
).to(device)
#打印模型的参数量
print("模型参数量：", sum(p.numel() for p in model.parameters())/1e6, "M")
```

或者这样
```python
model=GPT2Model(max_token_value=max_token_value,d_model=d_model,num_heads=num_heads,num_blocks=num_block,context_length=context_length).to(device)

x = torch.randint(0, max_token_value, (batch_size, context_length), dtype=torch.long).to(device).to(device)

torchinfo.summary(model,input_data=x)
```

Total params: 25,869,568
Trainable params: 25,869,568
Non-trainable params: 0

Input size (MB): 0.13
Forward/backward pass size (MB): 6878.66
Params size (MB): 103.48
Estimated Total Size (MB): 6982.27

### 开始训练 - 最初代码

```python
import torch
from torch.utils.data import DataLoader
from gpt2model import GPT2Model
from mydataset import PretrainDataset
from torch.optim import AdamW
from transformers import AutoTokenizer
from tqdm import tqdm

## 超参数定义

batch_size =64 # 每次训练的样本数
context_length=512# 上下文长度
d_model=512 # 嵌入向量的维度
num_block=8 # Transformer块数
num_heads=16 # 多头注意力的头数
max_token_value=6400
  
learning_rate=1e-4 # 学习率
dropout=0.1 # Dropout的比率
train_epoche=6 #在数据集上训练的轮数
eval_interval=50 # 评估的间隔
eval_iters=20
device = 'cuda:1' if torch.cuda.is_available() else 'cpu' # 设备
torch.manual_seed(1337) # 随机种子

  
tokenizer=AutoTokenizer.from_pretrained('/data1/zah_workspace/llmfromzero/tokenizer/model/minimind_tokenizer')
model=GPT2Model(max_token_value=max_token_value,d_model=d_model,num_heads=num_heads,num_blocks=num_block,context_length=context_length).to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
loss_fct=torch.nn.CrossEntropyLoss(reduction='none')#设置了 reduction='none'，这意味着它不会自动对损失进行平均或求和，而是返回每个样本的损失值。
## 数据集加载
dataset=PretrainDataset(data_dir='/data3/zah_work/datasets/pretrain_hq.jsonl',max_length=context_length,tokenizer=tokenizer)
  
mydataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
  
#训练模型

def train_epch(epoch):
    model.train()
    for step,(X,Y,lossmask) in tqdm(enumerate(mydataloader)):
        X=X.to(device)
        Y=Y.to(device)
        # print(X.shape)
        # print(Y.shape)

        lossmask=lossmask.to(device)
        logits=model(X)# logits 为(batchsize,contextlength,max_token_value) loss为)
        #掩码一下，去除padding
        losses = loss_fct(logits.view(-1, logits.size(-1)), Y.view(-1))
        #计算损失
        # print(f"loss shape: {loss.shape}")
        # print(f"lossmask shape: {lossmask.shape}")
        losses = losses.view(Y.shape)  # [1, 512]
        valid_tokens = lossmask.to(torch.bool)  # [1, 512]

        # 只计算有效token的损失
        valid_losses = losses[valid_tokens]
  
        # 对有效损失求平均

        final_loss = valid_losses.mean()
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        if(step%10==0):
            print(f"Loss (masked): {final_loss}")
        #
train_epch(10)
## 模型保存
```
主要就是加载Tokenizer、Model 设置优化器、损失函数、加载数据集
随后开始训练，打印损失

#### 
### 开始训练 - 优化版代码
让cursor进行分析,进行了如下
```
主要修改包括：

1. 降低了学习率从 1e-4 到 5e-5

2. 添加了学习率预热策略（warmup）

3. 添加了梯度裁剪（max_grad_norm = 1.0）

4. 增大了 batch size 从 64 到 128

5. 调整了 weight decay 从 0.001 到 0.01

6. 添加了 Xavier 初始化方法

7. 添加了学习率监控输出
```


```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

class FeedForwardNetWork(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()
        self.d_model=d_model
        self.d_ff=d_ff
        self.ffn=nn.Sequential(
            nn.Linear(in_features=self.d_model,out_features=self.d_ff),
            nn.ReLU(),
            nn.Linear(in_features=self.d_ff,out_features=self.d_model),
            nn.Dropout(0.1)
        )
        self.apply(init_weights)

    def forward(self,x):
        return self.ffn(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self,head_dim,context_length=16,dropout=0.1):
        super().__init__()
        self.head_dim=head_dim
        self.dropout=dropout
        self.context_length=context_length
        self.Wq=nn.Linear(head_dim,head_dim,bias=False)
        self.Wk=nn.Linear(head_dim,head_dim,bias=False)
        self.Wv=nn.Linear(head_dim,head_dim,bias=False)
        # apply mask
        self.register_buffer('mask',torch.tril(torch.ones(context_length,context_length)))
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self,x):
        B,T,C=x.shape
        assert T<=self.context_length
        assert C==self.head_dim
        Q=self.Wq(x)
        K=self.Wk(x)
        V=self.Wv(x)
        #计算分数
        attention=Q @ K.transpose(-2,-1) / math.sqrt(self.head_dim)
        # mask
        attention=attention.masked_fill(self.mask[:T,:T]==0,float('-inf'))
        attention_score=F.softmax(attention,dim=-1)
        attention_score = self.dropout_layer(attention_score)
        x=attention_score@V
        return x
  
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,context_length=16):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.context_length=context_length
        self.head_dim=d_model // num_heads

        self.heads=nn.ModuleList([ScaledDotProductAttention(head_dim=d_model//num_heads,context_length=context_length) for _ in range(num_heads)])
        self.projection_layer=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(0.1)

    def forward(self,x):
        # 将x拆分成多头
        batch_size=x.size(0)
        # view : 对数据重新分组，平铺后的矩阵数据位置还是一样的
        # transpose:x[a,b]的访问变为x[b,a]数据肯定是被打乱的了
        x=x.view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)
        heads_output=[]
        for i,head in enumerate(self.heads):
            heads_output.append(head(x[:,i]))
        heads_output=torch.cat(heads_output,dim=-1)
        out=self.projection_layer(heads_output)
        out=self.dropout(out)
        return out


class TransformerBlock(nn.Module):

    def __init__(self,d_model,num_heads,context_length=16):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.layer_norm1=nn.LayerNorm(d_model)
        self.layer_norm2=nn.LayerNorm(d_model)
        self.multi_head_attention =MultiHeadAttention(d_model=d_model,num_heads=num_heads,context_length=context_length)
        self.ff_network=FeedForwardNetWork(d_model=d_model,d_ff=4*d_model)

    def forward(self,x):
        multi_attention_output=self.multi_head_attention(x)
        x=self.layer_norm1(x+multi_attention_output)
        feed_forward_output=self.ff_network(x)
        x=self.layer_norm2(x+feed_forward_output)
        return x

class GPT2Model(nn.Module):
    def __init__(self,max_token_value,d_model,num_heads,num_blocks,context_length=16):
        super().__init__()
        self.d_model=d_model
        self.context_length=context_length
        self.token_embedding=nn.Embedding(max_token_value,d_model)
        self.transformer_block=nn.Sequential(*([TransformerBlock(d_model=d_model,num_heads=num_heads,context_length=context_length) for _ in range(num_blocks)]+[nn.LayerNorm(self.d_model)]))

  

        self.language_model_out_linear_layer = nn.Linear(in_features=self.d_model, out_features=max_token_value)
        self.apply(init_weights)

    def forward(self,idx,targets=None):
        B,T=idx.shape
        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        # change position_encoding_lookup_table from (context_length, d_model) to (T, d_model)
        position_embedding = position_encoding_lookup_table[:T, :].to(idx.device)
        x=self.token_embedding(idx)+position_embedding
        x=self.transformer_block(x)
        logits=self.language_model_out_linear_layer(x) # batch_size , time_stamp ,table
        return logits


    def generate(self,idx,max_new_tokens=100):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.context_length:]
            logits,loss=self(idx_crop)
            logist_last_timestep=logits[:,-1,:]
            probs=F.softmax(input=logist_last_timestep,dim=-1)
            idx_next=torch.multinomial(input=probs,num_samples=1)
            idx =torch.cat((idx,idx_next),dim=1)
        return idx
```

```python
import torch
from torch.utils.data import DataLoader
from gpt2model import GPT2Model
from mydataset import PretrainDataset
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
## 超参数定义
batch_size = 128 # 每次训练的样本数
context_length=512# 上下文长度
d_model=512 # 嵌入向量的维度
num_block=8 # Transformer块数
num_heads=16 # 多头注意力的头数
max_token_value=6400


learning_rate=5e-5 # 学习率
dropout=0.1 # Dropout的比率
train_epoche=6 #在数据集上训练的轮数
eval_interval=50 # 评估的间隔
eval_iters=20
warmup_steps = 1000  # 添加warmup步数
max_grad_norm = 1.0  # 梯度裁剪阈值
device = 'cuda:1' if torch.cuda.is_available() else 'cpu' # 设备
torch.manual_seed(1337) # 随机种子

tokenizer=AutoTokenizer.from_pretrained('/data1/zah_workspace/llmfromzero/tokenizer/model/minimind_tokenizer')
model=GPT2Model(max_token_value=max_token_value,d_model=d_model,num_heads=num_heads,num_blocks=num_block,context_length=context_length).to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
loss_fct=torch.nn.CrossEntropyLoss(reduction='none')#设置了 reduction='none'，这意味着它不会自动对损失进行平均或求和，而是返回每个样本的损失值。
## 数据集加载
dataset=PretrainDataset(data_dir='/data3/zah_work/datasets/pretrain_hq.jsonl',max_length=context_length,tokenizer=tokenizer)

mydataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

# 创建学习率调度器
num_training_steps = len(mydataloader) * train_epoche
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

#训练模型
def train_epch(epoch):
    model.train()
    for step,(X,Y,lossmask) in tqdm(enumerate(mydataloader)):
        X=X.to(device)
        Y=Y.to(device)
        # print(X.shape)
        # print(Y.shape)
        lossmask=lossmask.to(device)
        logits=model(X)# logits 为(batchsize,contextlength,max_token_value) loss为)
        #掩码一下，去除padding
        losses = loss_fct(logits.view(-1, logits.size(-1)), Y.view(-1))
        #计算损失
        # print(f"loss shape: {loss.shape}")
        # print(f"lossmask shape: {lossmask.shape}")
        losses = losses.view(Y.shape)  # [1, 512]
        valid_tokens = lossmask.to(torch.bool)  # [1, 512]
        # 只计算有效token的损失
        valid_losses = losses[valid_tokens]
  
        # 对有效损失求平均
        final_loss = valid_losses.mean()
        optimizer.zero_grad()
        final_loss.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # 更新学习率
        if(step%10==0):
            print(f"Loss (masked): {final_loss}")
            print(f"Learning rate: {scheduler.get_last_lr()[0]}")  # 打印当前学习率

train_epch(10)
## 模型保存
```
### 测试

训练到第二个epoch后loss在2.3~2.5震荡，直接停了。
```python
import torch
from torch.utils.data import DataLoader
from gpt2model import GPT2Model
from mydataset import PretrainDataset
from torch.optim import AdamW
from transformers import AutoTokenizer
from tqdm import tqdm
## 超参数定义
batch_size =64 # 每次训练的样本数
context_length=512# 上下文长度
d_model=512 # 嵌入向量的维度
num_block=8 # Transformer块数
num_heads=16 # 多头注意力的头数
max_token_value=6400
  
device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # 设备
tokenizer=AutoTokenizer.from_pretrained('./tokenizer/model/minimind_tokenizer')
model=GPT2Model(max_token_value=max_token_value,d_model=d_model,num_heads=num_heads,num_blocks=num_block,context_length=context_length).to(device)
model_path = './model/gpt2model_epoch3_iter72000_bs64.pt'
  
# 直接加载模型的state_dict
model.load_state_dict(torch.load(model_path, map_location=device))
  
idx=tokenizer("珠穆朗玛峰是",return_tensors='pt')
input_ids=idx['input_ids']
input_ids=input_ids.to(device)
newidx=model.generate(input_ids)
print(tokenizer.decode(newidx[0]))
```

输出: 
```json
珠穆朗玛峰是什么？玛玛峰是一个极坐线且势力的综合性。它洲是地球上最引力和最紧密的恒星之一，它们是一个巨大的恒星，位于赤道和非洲北部。这是一个惊艳中最强的恒星，位于太平洋中而已被永破碎时与太阳射线相互作用带的行星以及这种最强的恒星。</s> <s>好的，我马上回来，你能请问
```



有点人工智障了哈哈

后续路线

- 模型架构优化，提高训练速度
- 参数调整（炼丹） 重新训练
- 微调



---

文章参考:

博客地址: qwrdxer.github.io

欢迎交流: qq1944270374