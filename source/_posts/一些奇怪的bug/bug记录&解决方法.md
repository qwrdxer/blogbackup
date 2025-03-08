---
title: bug记录&解决方法
categories:
  - 一些奇怪的bug
date: 2023-10-02 16:05:12
tags:
---



## tokenizer = AutoTokenizer.from_pretrained("openai-gpt")  报错无法连接

```
We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like openai-gpt is not the path to a directory containing a file named config.json.
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
```

设置http代理



---

文章参考:

博客地址: qwrdxer.github.io

欢迎交流: qq1944270374