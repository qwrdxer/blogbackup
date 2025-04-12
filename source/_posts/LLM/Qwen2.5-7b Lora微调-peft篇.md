---
title: Qwen2.5-7b Lora微调-peft篇
categories:
  - LLM
date: 2025-04-10 14:32:36
tags:
---



### 参考链接

数据集 https://github.com/KMnO4-zx/huanhuan-chat/tree/master/dataset
教程 https://github.com/datawhalechina/self-llm/blob/master/models/Qwen2.5/05-Qwen2.5-7B-Instruct%20Lora%20.ipynb

参考了datawhale的教程，代码中加入了自己的理解
### 模型加载

使用transformers的`AutoTokenizer`, `AutoModelForCausalLM` 加载模型和Tokenizer
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
device="cuda:2" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model
```

```text
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(152064, 3584)
    (layers): ModuleList(
      (0-27): 28 x Qwen2DecoderLayer(
        (self_attn): Qwen2Attention(
          (q_proj): Linear(in_features=3584, out_features=3584, bias=True)
          (k_proj): Linear(in_features=3584, out_features=512, bias=True)
          (v_proj): Linear(in_features=3584, out_features=512, bias=True)
          (o_proj): Linear(in_features=3584, out_features=3584, bias=False)
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=3584, out_features=18944, bias=False)
          (up_proj): Linear(in_features=3584, out_features=18944, bias=False)
          (down_proj): Linear(in_features=18944, out_features=3584, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((3584,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=3584, out_features=152064, bias=False)
)
```

使用peft 的 `LoraConfig` 进行lora相关配置，使用 `get_peft_model` 将其嵌入模型中
```python
from peft import LoraConfig ,TaskType,get_peft_model
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    r=8,
    lora_alpha=32, # lora的权重影响
    lora_dropout=0.1,
)

model=get_peft_model(model,lora_config)
model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
```

| 情况                               | 公式                     | 影响                                                         |
| ---------------------------------- | ------------------------ | ------------------------------------------------------------ |
| **`alpha = r`**                    | h(x)=Wx+BAxh(x)=Wx+BAx   | LoRA 更新直接相加，无额外缩放（默认行为）。                  |
| **`alpha = 2r`**                   | h(x)=Wx+2BAxh(x)=Wx+2BAx | LoRA 更新放大 2 倍，适配器影响更强（论文推荐）。             |
| **`alpha = 32, r=8`**              | h(x)=Wx+4BAxh(x)=Wx+4BAx | 你的配置（`alpha / r = 4`），LoRA 影响较大，适合强任务适配。 |
| 可以打印一下 trainable为可训练参数 |                          |                                                              |
```python
model.print_trainable_parameters()
#trainable params: 20,185,088 || all params: 7,635,801,600 || trainable%: 0.2643
model.device
#device(type='cuda', index=2)
```

### 数据处理

数据集在这里下载https://github.com/KMnO4-zx/huanhuan-chat/tree/master/dataset
使用Dataset加载数据，通过传入函数对原始数据进行统一处理、

```python
from datasets import Dataset
import pandas as pd
#数据集加载
df=pd.read_json('huanhuan.json')
ds=Dataset.from_pandas(df)
ds[:1]
```

```json
{'instruction': ['小姐，别的秀女都在求中选，唯有咱们小姐想被撂牌子，菩萨一定记得真真儿的——'], 'input': [''], 'output': ['嘘——都说许愿说破是不灵的。']}
```

然后是对原始数据集进行处理，处理的方式为: insturction+Input为用户输入，output为模型输出
```python
def process_func(example):
    MAX_LENGTH=384
    input_ids,attention_mask,labels=[],[],[]
    instruction = tokenizer(f"<|im_start|>system\n现在你要扮演皇帝身边的女人--甄嬛<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response=tokenizer(f"{example['output']}",add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
```

1. insturction 构造的是用户的输入，这部分并不是模型生成的。
2. Response是应该由模型生成的
3. 我们微调的目标是让模型根据人工输入（instruction） 预测出相应的答案 （Response）
4. 通过labels，将instruction部分掩去（设置为-100）
5. -100， PyTorch的`CrossEntropyLoss`默认将`ignore_index=-100`，因此用`-100`标记的位置会被自动排除在损失计算外。

开始对原始数据进行处理
```python
#生成 input_ids attention_mask labels 列, 移除原始列
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

```
Map: 100%|██████████| 3729/3729 [00:01<00:00, 2586.51 examples/s]



### 开始训练

```python
from transformers import  TrainingArguments,Trainer,DataCollatorForSeq2Seq
```

配置训练参数
```python
args=TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4, #**累积 4 个批次的梯度**后进行一次参数更新（等效于 `batch_size=16`）。
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    gradient_checkpointing=True,
    save_on_each_node=True,
)
```

构建训练器
```python
trainer=Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    tokenizer=tokenizer,
    #1. 动态填充同一批次内的样本到相同长度。
    #2. 生成注意力掩码（`attention_mask`），标记填充位置（避免模型计算这些位置）。
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
```

```开始训练
trainer.train()
```

### 结果测试
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
mode_path = '/root/autodl-tmp/qwen/Qwen2.5-7B-Instruct/'
lora_path = './output/Qwen2.5_instruct_lora/checkpoint-10' # 这里改称你的 lora 输出对应 checkpoint 地址
# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)
# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载lora权重 ,可以注释掉这一行看看原始模型的输出
model = PeftModel.from_pretrained(model, model_id=lora_path)
prompt = "你是谁？"
inputs = tokenizer.apply_chat_template([{"role": "user", "content": "假设你是皇帝身边的女人--甄嬛。"},{"role": "user", "content": prompt}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       ).to('cuda')

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

lora_model_output:
我是甄嬛，家父是大理寺少卿甄远道。

qwen_output:
我是您的助手Qwen，由阿里云开发的人工智能模型。虽然我不能是《甄嬛传》中的甄嬛，但我可以和您分享关于这部电视剧的信息，或者以一种虚拟的方式体验一下那个角色的生活，当然这都是基于想象和创作。您想了解些什么呢？

### 将lora合并到原始模型中

通过merg_and_unload()来获取一个合并后的模型，将其保存

```python
merged_model = model.merge_and_unload()  # 关键步骤！
# 保存合并后的完整模型
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

### 使用SwanLab记录结果

[SwanLab](https://github.com/swanhubx/swanlab) 是一个开源的模型训练记录工具，面向AI研究者，提供了训练可视化、自动日志记录、超参数记录、实验对比、多人协同等功能。在SwanLab上，研究者能基于直观的可视化图表发现训练问题，对比多个实验找到研究灵感，并通过在线链接的分享与基于组织的多人协同训练，打破团队沟通的壁垒。

SwanLab与Transformers已经做好了集成，用法是在Trainer的`callbacks`参数中添加`SwanLabCallback`实例，就可以自动记录超参数和训练指标，简化代码如下：

```python
from swanlab.integration.transformers import SwanLabCallback
from transformers import Trainer

swanlab_callback = SwanLabCallback()

trainer = Trainer(
    ...
    callbacks=[swanlab_callback],
)
```

准备测试数据
```python
test_df = pd.read_json('huanhuan.json')[:5]
```

编写一个预测函数
```python

def predict(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

```

编写callback

```python
from swanlab.integration.transformers import SwanLabCallback
import swanlab

class HuanhuanSwanLabCallback(SwanLabCallback):  
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model, **kwargs)
        print("训练开始")
        print("未开始微调，先取3条主观评测：")
        test_text_list = []
        for index, row in test_df[:3].iterrows():
            input_value = row["instruction"]

            messages = [
                {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
                {"role": "user", "content": f"{input_value}"},
            ]

            response = predict(messages, peft_model, tokenizer)
            messages.append({"role": "assistant", "content": f"{response}"})
            result_text = f"【Q】{messages[1]['content']}\n【LLM】{messages[2]['content']}\n"
            print(result_text)
            test_text_list.append(swanlab.Text(result_text, caption=response))
        swanlab.log({"Prediction": test_text_list}, step=0)

    def on_epoch_end(self, args, state, control, **kwargs):
        # ===================测试阶段======================
        test_text_list = []
        for index, row in test_df.iterrows():
            input_value = row["instruction"]
            ground_truth = row["output"]
  

            messages = [
                {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
                {"role": "user", "content": f"{input_value}"},
            ]

  
            response = predict(messages, peft_model, tokenizer)
            messages.append({"role": "assistant", "content": f"{response}"})
            if index == 0:
                print("epoch", round(state.epoch), "主观评测：")
            result_text = f"【Q】{messages[1]['content']}\n【LLM】{messages[2]['content']}\n【GT】 {ground_truth}"
            print(result_text)
            test_text_list.append(swanlab.Text(result_text, caption=response))


        swanlab.log({"Prediction": test_text_list}, step=round(state.epoch))
```


创建callback并使用
```python
swanlab_callback = HuanhuanSwanLabCallback(
    project="Qwen2.5-LoRA-Law",
    experiment_name="7b",
    config={
        "model": "https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct",
        "system_prompt": "现在你要扮演皇帝身边的女人--甄嬛",
        "lora_rank": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1
    },
)


trainer=Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],

)
```

开始训练
```python
trainer.train()
```
### 问题记录

####  Trainer将模型移入 cuda:0

- PyTorch 和 Hugging Face 的默认行为是优先使用 `cuda:0`，除非显式干预。
- 分布式训练时，`Trainer` 会通过 `torch.distributed` 分配设备，但单机情况下需手动指定。



---

文章参考:

博客地址: qwrdxer.github.io

欢迎交流: qq1944270374