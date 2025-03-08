---
title: 大模型论文
categories:
  - 论文阅读
  - 人工智能
tags:
  - GPT
  - LLM
date: 2023-10-01 22:17:21
---





# GPT

## 1.1论文:

https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf

## 1.2概述:

1. pre-training ,即对模型使用大量的未标注的文本预料进行训练,使其能够拥有强大的自然语言理解能力
2. fine-tuning ，根据具体的任务，使用处理好的结构化数据进行训练，来增加其对特定任务的处理能力

## 1.3代码:

https://huggingface.co/docs/transformers/model_doc/openai-gpt





---

| 模型名称                 | 时间    | 是否开源 | 参数规模  | Paper                                                        | Code                                                         |
| ------------------------ | ------- | -------- | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| GPT                      | 2018-06 | 是       | 117M      | [Paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) | [Hugging Face](https://huggingface.co/docs/transformers/model_doc/openai-gpt) |
| GPT-2                    | 2019-02 | 是       | 150M-1.5B | [Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | [Hugging Face](https://huggingface.co/docs/transformers/model_doc/gpt2) |
| GPT-3                    | 2020-05 | 否       | 125M-175B | [Wiki](https://en.wikipedia.org/wiki/GPT-3) [Arxiv](https://arxiv.org/abs/2005.14165) | -                                                            |
| GPT-3.5 (InstructionGPT) | 2022-01 | 否       | 175B      | [Blog](https://openai.com/research/instruction-following)    | -                                                            |
| GPT-4                    | 2023-03 | 否       | 未知      | [Blog](https://openai.com/research/instruction-following) [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) | -                                                            |

# Bert-paper

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding













----



以下是GPT和BERT之间的主要关系和区别：

1. **预训练目标**：
   - GPT：GPT模型采用了单向语言模型（unidirectional language modeling）的预训练目标。在预训练期间，GPT模型尝试根据上下文生成下一个单词，以捕获语言的统计特性和上下文理解。
   - BERT：BERT采用了双向语言模型（bidirectional language modeling）的预训练目标。在BERT中，输入文本的一半被掩盖，模型需要预测掩盖位置的单词，这使得模型能够同时考虑上下文的左侧和右侧。
2. **模型架构**：
   - GPT：GPT模型通常是解码器-only架构，只包括解码器层。它在生成任务中表现出色，如文本生成、对话系统等。
   - BERT：BERT模型包括编码器层和解码器层，但通常只使用编码器层来进行特征提取。BERT的主要任务是生成上下文相关的嵌入，通常用于下游任务的微调。
3. **应用领域**：
   - GPT：GPT模型更适合生成文本，因此在生成性任务中表现出色，如对话生成、文本摘要、文章创作等。
   - BERT：BERT模型的主要优势在于上下文相关的表示，因此在下游NLP任务中表现出色，如文本分类、命名实体识别、语义相似性等。
4. **模型大小**：
   - GPT和BERT都有不同规模的变体，可以根据任务和计算资源进行选择。GPT-3等大型GPT变体拥有数十亿甚至上百亿个参数，而BERT的变体通常规模较小。

总之，GPT和BERT都是基于Transformer架构的强大NLP模型，它们在预训练和应用中有不同的方法和目标。选择使用哪个模型取决于您的具体任务和需求，以及您可用的计算资源。有时候，研究人员和工程师还会将它们的优势结合起来，例如在BERT的基础上进行GPT-style的微调，以获得更好的性能。



文章参考:

> https://github.com/WangHuiNEU/llm#--foundation-model-------
>
> 
>
> https://blog.csdn.net/yangfengling1023/article/details/85054871

博客地址: qwrdxer.github.io

欢迎交流: qq1944270374