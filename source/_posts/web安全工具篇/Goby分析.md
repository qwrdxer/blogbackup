---
title: Goby分析
toc: true
categories:
  - web安全工具篇
date: 2022-01-09 13:35:49
tags:
---



Wireshark抓包

tcp.port==xxxx and ip.addr==127.0.0.1

 ![image-20220109133714265](Goby%E5%88%86%E6%9E%90/image-20220109133714265.png)

启动goby

​                               

Goby.exe为前端程序(electron)

Goby-cmd.exe为后端程序（go）绑定端口为8361，提供http服务、漏洞扫描服务

 

一开始为前端与后端建立http连接,随后切换为websocket进行通信， 前两次绑定失败，应该是后端服务在启动，最后绑定32441端口 ，该端口向前端发送具体的扫描结果，如开放端口信息、指纹、漏洞等

 *<!-- more -->* 



 ![image-20220109133729548](Goby%E5%88%86%E6%9E%90/image-20220109133729548.png)

获取环境变量

 ![image-20220109133734228](Goby%E5%88%86%E6%9E%90/image-20220109133734228.png)

获取执行完毕的任务

 ![image-20220109133738043](Goby%E5%88%86%E6%9E%90/image-20220109133738043.png)

获取poc列表

 ![image-20220109133741853](Goby%E5%88%86%E6%9E%90/image-20220109133741853.png)

 

开启一个扫描任务后，继续分析

点击开启后，前端post请求/api/v1/startscan/ 到后端进行扫描 ，原端口为18731

 ![image-20220109133749036](Goby%E5%88%86%E6%9E%90/image-20220109133749036.png)

这次请求只用来提交任务，提交完毕后就连接关闭了

 ![image-20220109133752820](Goby%E5%88%86%E6%9E%90/image-20220109133752820.png)

 

火绒剑，马赛克部分为扫描目标，不作考虑

 ![image-20220109133756569](Goby%E5%88%86%E6%9E%90/image-20220109133756569.png)

32441端口，用于后端向前端传递目标端口协议、指纹信息。

 ![image-20220109133800195](Goby%E5%88%86%E6%9E%90/image-20220109133800195.png)

对应于前端的这两块部分

 

 ![image-20220109133803737](Goby%E5%88%86%E6%9E%90/image-20220109133803737.png)

 

 

 

 

 

 ![image-20220109133809157](Goby%E5%88%86%E6%9E%90/image-20220109133809157.png)

 

前端一开始会不断调用

````shell
/api/v1/getProgress 

/api/v1/getStatisticsData 
````



 

这两个接口，getsatisticData获取 指纹、ico图标等扫描结果信息

 

 

Getprogress用来判断这个任务是否完成

 ![image-20220109133826156](Goby%E5%88%86%E6%9E%90/image-20220109133826156.png)

其中progress的值对应前端的扫描百分数 

 ![image-20220109133830660](Goby%E5%88%86%E6%9E%90/image-20220109133830660.png)

 

扫描完毕，（getprogress结果为100）

调用如下接口进行前端数据展示



![image-20220109133836262](Goby%E5%88%86%E6%9E%90/image-20220109133836262.png)

 ![image-20220109133848210](Goby%E5%88%86%E6%9E%90/image-20220109133848210.png)

 

 

