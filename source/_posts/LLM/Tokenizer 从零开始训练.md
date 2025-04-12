---
title: Tokenizer 从零开始训练
categories:
  - LLM
date: 2025-04-06 20:50:21
tags:
---



### 资料参考

视频
https://www.bilibili.com/video/BV13Ci6YYEUY/
视频对应代码仓库
https://github.com/wyf3/llm_related

中文数据集
https://huggingface.co/datasets/TurboPascal/tokenizers_example_zh_en/tree/main

原理分析
https://huggingface.co/docs/transformers/tokenizer_summary
### 上手训练
**下载数据集并查看**
```shell
tail -n 10 data.txt 
```


	庄贾 (秦朝) \n 庄贾，秦朝末年军事人物，陈胜、吴广发动大泽之变后，担任陈胜的车夫。 \n 秦二世元年（前209年），秦少府章邯率军进攻陈胜的上柱国蔡赐，斩之。又进攻陈县（今河南淮阳），陈胜亲自应战，却接连败退，逃至下城父（今安徽涡阳），庄贾趁乱刺杀了陈胜，投降秦兵。后吕臣起兵为陈胜报仇，杀了庄贾。
	余爱国 \n 余爱国，湖南岳阳人。1976年2月加入中国共产党。湖南教育学院（现并入湖南师范大学）政治专业函授本科毕业，香港大学工商管理函授研究生学历。 \n 历任岳阳长岭炼油厂兴长公司党委副书记、书记，长岭炼油化工厂党委副书记；岳阳市人民政府副市长、市委常委；湘潭市委副书记、市纪委书记等职。2007年1月，任湘潭市市长。2010年1月，任湖南省人民政府副秘书长。2014年1月，不再担任湖南省人民政府副秘书长。
	高雄市皮影戏馆 \n 高雄市皮影戏馆，位于高雄市冈山区文化中心内，隶属于高雄市立历史博物馆，是全台唯一的皮影戏馆，每个月第二个礼拜日都会有固定的皮偶戏团在此演出，让喜爱皮影戏的民众们可以好好回味这传统艺术的文化。 \n 皮影戏从两百多年前的冈山镇发源，在馆内展出相当多早期表演的传统乐器与一系列具年代精致的皮影戏偶，更有皮偶实做区提供游客们现场参与雕刻与操玩。专题馆中则介绍了现今台湾的五个皮影剧团的相关历史与发展现况，喜欢皮影戏的观众们可在此找回这传统的艺术文化并透过此继续传承。 \n 皮影戏馆于1994年3月13日正式开馆营运，除了展示教育与典藏的传统功能之外，皮影戏馆也具备推广与研究的功能。2010年台风凡那比风灾导致皮影戏馆建筑体受损严重，闭馆整修后，于2013年3月竣工重新开馆，并隶属于高雄市立历史博物馆所属场馆。 \n 本馆分为｢传习教室｣、｢主题展示馆｣、｢资源中心｣、｢剧场｣、｢数位剧院｣、｢体验区｣等六大主题区。
	"长柄芥属 \n 长柄芥属（学名：""Macropodium""）是十字花科下的一个属，为多年生草本植物。该属共有2种，分布于中亚、萨哈林岛、日本北部。"
	"鞘花属 \n 鞘花属（学名：""Macrosolen""）是桑寄生科下的一个属。该属共有约40种，分布于亚洲南部和东南部。"
	"硬皮豆属 \n 硬皮豆属（学名：""Macrotyloma""）是蝶形花科下的一个属，为攀援、匍匐或直立草本植物。该属共有约25种，分布于非洲和亚洲。"
	"紫荆木属 \n 紫荆木属（学名：""Madhuca""）是山榄科下的一个属，为乔木植物。该属共有约85种，分布于印度、马来西亚。"
	"十大功劳属 \n 十大功劳属（学名：""Mahonia""）是小檗科下的一个属。该属共有约100种，分布于美洲中部和北部及亚洲。 \n 本属大部分的物种都可入药，在中药学中包含根、茎、叶等器官都可作为药材，而药效视物种而有所不同，据说有十种疗效，因而得名。"
	"舞鹤草属 \n 舞鹤草属（学名：""Maianthemum""）是百合科下的一个属，为多年生、矮小草本植物。该属共有4种，分布于北温带。 \n 最新资料把该属列为天门冬科植物"
	"牛筋藤属 \n 牛筋藤属（学名：""Trophis""，异名""Malaisia""）是桑科下的一个属，为藤本植物。该属仅有牛筋藤（""Trophi scandens""）一种，分布于东南亚和大洋洲。"


**导入对应的包**
```python
import random
import json
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
import os
random.seed(42)
```

**创建Trainner**
```python
tokenizer = Tokenizer(models.BPE()) #使用BPE进行训练
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)# 预处理的词频统计，按照字节进行统计中文字符会被划分为三个字节 BBPE

# 定义特殊token
special_tokens = ["<unk>", "<s>", "</s>"]

# 设置训练器并添加特殊token
trainer = trainers.BpeTrainer(
	vocab_size=6400,#最终的词表大小为6400
	special_tokens=special_tokens,  # 确保这三个token被包含
	show_progress=True,
	initial_alphabet=pre_tokenizers.ByteLevel.alphabet()#初始的词表为
)

```

- 初始的词表大小为256 + 3
- 最终词表大小为6400
- `special_tokens` 是添加**完整的、不可拆分的字符串 token**，但和 alphabet 无关，它们不会参与 BPE 合并


**开始训练**
```python
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
```

![image-20250406205046570](Tokenizer%20%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E8%AE%AD%E7%BB%83/image-20250406205046570.png)

Pre-processing :在此阶段，代码可能执行一些初步的数据清理和格式化操作，以便后续的处理。

Tokenize words:由于配置了`pre_tokenizers.ByteLevel()`，因此，文本会被按照字节级别分割。这意味着，即使是中文文本，也会被分割成单个字节

Count pairs: 此步骤计算所有字节对在训练数据中出现的次数。因为上一步骤已经按照字节分割了所有的文本，所以此处统计的是字节对的频率。

Compute merges:根据词对的统计频率，BPE算法迭代地将最常见的字节对合并成新的词元。这个过程会持续进行，直到词汇表达到预设的大小。

**保存文件**

```python
   # 设置解码器
tokenizer.decoder = decoders.ByteLevel()
# 检查特殊token的索引
assert tokenizer.token_to_id("<unk>") == 0
assert tokenizer.token_to_id("<s>") == 1
assert tokenizer.token_to_id("</s>") == 2
# 保存tokenizer
tokenizer_dir = "./model/minimind_tokenizer"
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
tokenizer.model.save("./model/minimind_tokenizer")
```

`tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))` 这个文件保存了分词器的完整配置信息

`tokenizer.model.save("../model/minimind_tokenizer")` 创建文件： `merges.txt` 和 `vocab.json`


- `tokenizer.json` 是一个更全面的配置文件，适用于 `transformers` 库。
- `merges.txt` 和 `vocab.json` 是 `tokenizers` 库本身使用的文件格式。

- `merges.txt`：
    - 保存 BPE 算法中的词元合并规则，每一行表示一个合并操作。
- `vocab.json`：
    - 保存词汇表，即词元到 ID 的映射。


```python
    # 手动创建配置文件

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True

            },
            "1": {
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },

        "additional_special_tokens": [],
        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<unk>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<s>system\\n' + system_message + '</s>\\n' }}{% else %}{{ '<s>system\\n你是 MiniMind，是一个有用的人工智能助手。</s>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)
```
这个 `tokenizer_config.json` 文件用于配置 `transformers` 库中的分词器，特别是 `PreTrainedTokenizerFast` 类。

**加载并使用**

```python
from transformers import AutoTokenizer
tokenizer_dir = "./tokenizer/model/minimind_tokenizer"  # 替换为您的分词器目录路径
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

text = "<s>这是一个示例文本。</s>"
inputs = tokenizer(text, return_tensors="pt")  # 将文本转换为 PyTorch 张量
print(inputs)
# 使用分词器进行解码
token_ids = inputs["input_ids"][0]
decoded_text = tokenizer.decode(token_ids)
print("解码结果：", decoded_text)
```

```cmd
{'input_ids': tensor([[ 1, 647, 3118, 1182, 2154, 677, 624, 274, 2]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])}

解码结果： <s>这是一个示例文本。</s>
```

可以加载对话格式的数据
```python
# 使用chat_template进行对话格式化。

messages = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，有什么我可以帮助你的吗？"},
    {"role": "user", "content": "今天天气怎么样？"},
]
chat_input = tokenizer.apply_chat_template(messages, tokenize=False)
print("chat_template格式化结果：", chat_input)
```

### 原理分析

#### Tokenizer的作用
我们可以通过在线网站来迅速体验
今天天气昨天天气 -> [10941, 1487, 25896, 57563, 1487, 25896]

![image-20250406205059633](Tokenizer%20%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E8%AE%AD%E7%BB%83/image-20250406205059633.png)



![image-20250406205111922](Tokenizer%20%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E8%AE%AD%E7%BB%83/image-20250406205111922.png)



我们可以迅速得出两个结论:

- Tokenizer负责将文本转换成独特的整型数字序列。
- 字符数量跟token数量并不一一对应。





#### Word-base Tokenizers
让我们以这句话为例子
"Don't you love 🤗 Transformers? We sure do."
对其划分的最简单方式是按照空格拆分
-> ["Don't", "you", "love", "🤗", "Transformers?", "We", "sure", "do."]

观察结果，可以对标点符号进行拆分
-> ["Don", "'", "t", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]

[spaCy](https://spacy.io/) 和 [Moses](http://www.statmt.org/moses/?n=Development.GetStarted) 是两种流行的基于规则的分词器。在我们的示例中应用它们，_spaCy_ 和 _Moses_ 将输出如下内容：
-> ["Do", "n't", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]

可以看出，这里使用了空格和标点符号分词，以及基于规则的分词化,这种方法很简单，但是会导致最终词汇表大小过大。例如 ，[Transformer XL](https://huggingface.co/docs/transformers/model_doc/transfo-xl) 使用空格和标点符号分词，导致词汇大小为 267,735！


#### subword-based Tokenizers
子词分词算法基于以下原则：经常使用的词不应拆分为较小的子词，而应将稀有词分解为有意义的子词。

例如 `annoyingly` 可能被认为是一个稀有词，可以分解为 `annoying` 和`ly`,  `annoying` 和 `ly` 作为独立的子词会出现得更频繁，同时`annoyingly` 的含义则由 `annoying` 和 `ly` 的复合含义保留.

**Byte-Pair Encoding (BPE)字节对编码**
BPE算法包含两个部分 "词频统计" 与”词表合并“

- 词频统计(pre-tokenization) 可以采用word-based tokennization,词频统计后，从训练数据中获取了唯一单词集，并确定了每一个词在训练数据中出现的频率。
- 然后BPE 创建一个由唯一单词集中出现的所有符号组成的基本词汇表，并学习合并规则以从基本词汇表的两个符号中形成一个新符号。
例如，我们假设在预分词化后，已经确定了以下一组单词（包括它们的频率）：
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)

因此，基本词汇表是 `["b", "g", "h", "n", "p", "s", "u"]` 。

拆分唯一单词集
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)

然后BPE会计算每个可能的符号对的频率，并选择出现率最高的符号对
hu:15 ,ug:20, pu:17,un:16.bu:4,gs:5

ug被选择，构建新的基本词表`["b", "g", "h", "n", "p", "s", "u", "ug"]`

唯一单词集合也进行更新:
("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)

继续计算相邻token对出现的频率
hug:15 ,pug:5, pu:12 ,un:16 ,bu:4 ,ugs:5

un被选择，构建新的基本词表`["b", "g", "h", "n", "p", "s", "u", "ug","un"]`

唯一单词集合也进行更新:
("h" “ug", 10),("p" "ug", 5),("p" "un", 12),("b" "un",4),("h" "ug" "s",5)

计算频率
hug:15 ...
hug被选择,构建新的基本词表`["b", "g", "h", "n", "p", "s", "u", "ug","un","hug"]`

...总之，通过多轮更新，最终获取到一个词表（BPE的合并次数是一个超参数）

**Byte-level BPE**

BPE的缺点
- 包含所有可能的基本字符的基本词汇表可能相当大
- 如，所有的unicode字符都被视为基本字符(如中文)

改进: Byte-level BPE
- 将字节（byte)视为基本token
- 两个字节合并即可以表示unicode

GPT-2使用字节作为基本词汇表，这是一个聪明的技巧，可以强制基本词汇表的大小为 256，同时确保每个基本字符都包含在词汇表中。通过一些额外的规则来处理标点符号，GPT2 的分词器可以在不需要符号的情况下对每个文本进行分词 。[GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt) 的词汇量为 50,257，对应于 256 字节的基本标记、特殊的文本结束标记和通过 50,000 次合并学习的符号。

我们训练使用的就是BBPE



---

文章参考:

博客地址: qwrdxer.github.io

欢迎交流: qq1944270374