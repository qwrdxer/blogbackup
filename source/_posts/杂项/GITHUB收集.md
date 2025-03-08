---
title: GITHUB收集
toc: true
categories:
  - 杂项
date: 2022-01-12 20:07:03
tags:
---

## scan

### https://github.com/lcvvvv/gonmap

gonmap是一个go语言的nmap端口扫描库，使用nmap开源的端口扫描策略，在效率和速度上取得一个折中的中间值，便于做大网络环境的资产测绘。



#### webanalyze指纹识别

https://github.com/rverton/webanalyze

#### 水泽信息收集自动化工具

https://github.com/0x727/ShuiZe_0x727

#### 基于YAML语法模板的定制化快速漏洞扫描器

Nuclei使用零误报的定制模板向目标发送请求，同时可以对大量主机进行快速扫描。Nuclei提供TCP、DNS、HTTP、FILE等各类协议的扫描，通过强大且灵活的模板，可以使用Nuclei模拟各种安全检查。

https://github.com/projectdiscovery/nuclei/

https://dhiyaneshgeek.github.io/web/security/2021/07/19/hack-with-automation/

#### Vulmap 是一款 web 漏洞扫描和验证工具, 可对 webapps 进行漏洞扫描, 并且具备漏洞验证功能

https://github.com/zhzyker/vulmap

#### Asset discovery and identification tools 快速识别 Web 指纹信息，定位资产类型。辅助红队快速定位目标资产信息，辅助蓝队发现疑似脆弱点

https://github.com/zhzyker/dismap

#### crlfuzz   A fast tool to scan CRLF vulnerability written in Go

https://github.com/dwisiswant0/crlfuzz

#### fscan  内网扫描器

https://github.com/shadow1ng/fscan

*<!-- more -->* 

#### 如何编写一个xray POC

https://zhuanlan.zhihu.com/p/78334648

#### DorkScout

可以通过Google搜索引擎自动查找互联网上存在安全漏洞的应用程序或机密文件，DorkScout首先会从https://www.exploit-db.com/google-hacking-database获取可访问到的Dock列表，然后它会扫描一个给定的目标，或扫描所有获取到的Dock。

https://github.com/R4yGM/dorkscout

#### nucei

通过yaml模板对目标进行扫描

https://github.com/projectdiscovery/nuclei

yaml 漏洞模板集合

https://github.com/projectdiscovery/nuclei-templates

#### Web-Attack-Cheat-Sheet一份 Web 攻击速查表，里面提供了相关开发工具与实现方案

速查表覆盖了 DNS 和 HTTP 检测、视觉识别、静态应用安全测试、漏洞搜索、SQL 注入、SSRF（服务器端请求伪造）等技术点。

https://github.com/riramar/Web-Attack-Cheat-Sheet

#### Finger  一款红队在大量的资产中存活探测与重点攻击系统指纹探测工具

https://github.com/EASY233/Finger

#### TideFinger

https://github.com/TideSec/TideFinger

#### 360/0Kee-Team/crawlergo动态爬虫结合长亭XRAY扫描器的被动扫描功能

https://github.com/timwhitez/crawlergo_x_XRAY

#### DalFox(Finder Of XSS) / Parameter Analysis and XSS Scanning tool based on golang

https://github.com/hahwul/dalfox

#### 一款功能强大的漏洞扫描器

子域名爆破使用aioDNS，asyncio异步快速扫描，覆盖目标全方位资产进行批量漏洞扫描，中间件信息收集，自动收集ip代理，探测Waf信息时自动使用来保护本机真实Ip，在本机Ip被Waf杀死后，自动切换代理Ip进行扫描，Waf信息收集(国内外100+款waf信息)包括安全狗，云锁，阿里云，云盾，腾讯云等，提供部分已知waf bypass 方案，中间件漏洞检测(Thinkphp,weblogic等 CVE-2018-5955,CVE-2018-12613,CVE-2018-11759等)，支持SQL注入, XSS, 命令执行,文件包含, ssrf 漏洞扫描, 支持自定义漏洞邮箱推送功能

https://github.com/YagamiiLight/Cerberus

#### crawlergo是一个使用chrome headless模式进行URL收集的浏览器爬虫

对整个网页的关键位置与DOM渲染阶段进行HOOK，自动进行表单填充并提交，配合智能的JS事件触发，尽可能的收集网站暴露出的入口。内置URL去重模块，过滤掉了大量伪静态URL，对于大型网站仍保持较快的解析与抓取速度，最后得到高质量的请求结果集合。

crawlergo 目前支持以下特性：

- 原生浏览器环境，协程池调度任务
- 表单智能填充、自动化提交
- 完整DOM事件收集，自动化触发
- 智能URL去重，去掉大部分的重复请求
- 全面分析收集，包括javascript文件内容、页面注释、robots.txt文件和常见路径Fuzz
- 支持Host绑定，自动添加Referer
- 支持请求代理，支持爬虫结果主动推送

https://github.com/Qianlitp/crawlergo

https://github.com/ExpLangcn/WanLi

使用Dirsearch, Subfinder, Ksubdomain, Httpx、nuclei工具进行快速目标资产检查并对目标资产进行敏感文件、敏感路径、漏洞验证检测。

## monitor

#### Medusa是一个红队武器库平台，

目前包括XSS平台、协同平台、CVE监控、免杀生成、DNSLOG、钓鱼邮件、文件获取等功能，持续开发中

https://github.com/Ascotbe/Medusa

#### ARL(Asset Reconnaissance Lighthouse)资产侦察灯塔系统

旨在快速侦察与目标关联的互联网资产，构建基础资产信息库。 协助甲方安全团队或者渗透测试人员有效侦察和检索资产，发现存在的薄弱点和攻击面。

https://github.com/TophantTechnology/ARL

#### 实时监控github上新增的cve和安全工具更新，多渠道推送通知

https://github.com/yhy0/github-cve-monitor



#### https://github.com/Le0nsec/SecCrawler

一个方便安全研究人员获取每日安全日报的爬虫和推送程序，目前爬取范围包括先知社区、安全客、Seebug Paper、跳跳糖、奇安信攻防社区、棱角社区，持续更新中。

## develop

#### Go语言学习资料

https://github.com/yangwenmai/learning-golang

## teach

#### 渗透步骤，web安全，CTF，业务安全，人工智能，区块链安全，安全开发，无线安全，社会工程学，二进制安全，移动安全，红蓝对抗，运维安全，风控安全，linux安全

https://github.com/Ascotbe/HackerMind

#### 各种安全相关思维导图整理收集

https://github.com/phith0n/Mind-Map

#### 正则表达式学习

https://regexlearn.com/learn

#### 平常看到好的渗透hacking工具和多领域效率工具的集合

https://github.com/taielab/awesome-hacking-lists

#### 内网渗透教程

https://github.com/Ridter/Intranet_Penetration_Tips

#### 人工智能教程

https://github.com/microsoft/AI-System

#### 365天获取玄武实验室的工作

https://github.com/Vancir/365-days-get-xuanwulab-job

#### Java安全相关的漏洞和技术demo

https://github.com/threedr3am/learnjavabug

#### 从零开始内网渗透学习

https://github.com/l3m0n/pentest_study

#### ffffffff0x 团队维护的安全知识框架,内容包括不仅限于 web安全、工控安全、取证、应急、蓝队设施部署、后渗透、Linux安全、各类靶机writup

https://github.com/ffffffff0x/1earn

#### 网络安全+AI

https://github.com/jivoi/awesome-ml-for-cybersecurity

#### 安全思维导图集合

https://github.com/SecWiki/sec-chart

## plugin

#### cel

轻松的进行逻辑运算

https://github.com/google/cel-spec

#### 扫描器Awvs 11和Nessus 7 Api利用脚本

https://github.com/se55i0n/Awvs_Nessus_Scanner_API

#### BugRepoter_0x727(自动化编写报告平台)根据安全团队定制化协同管理项目安全，可快速查找历史漏洞，批量导出报告。

https://github.com/0x727/BugRepoter_0x727

#### 调用 chrome 对指定url截图

https://github.com/sensepost/gowitness

#### FOFA Pro view 是一款FOFA Pro 资产展示浏览器插件，目前兼容 Chrome、Firefox、Opera。

https://github.com/fofapro/fofa_view

#### cobaltstrike的相关资源汇总

https://github.com/zer0yu/Awesome-CobaltStrike

## exploit

#### webcrack

WebCrack是一款web后台弱口令/万能密码批量检测工具，在工具中导入后台地址即可进行自动化检测。

https://github.com/yzddmr6/WebCrack

#### Intranet pentesting tool with webui 开源图形化内网渗透工具

https://github.com/FunnyWolf/Viper

#### A proof-of-concept tool for generating payloads that exploit unsafe Java object deserialization.

https://github.com/frohoff/ysoserial

#### 一款适用于以HW行动/红队/渗透测试团队为场景的移动端(Android、iOS、WEB、H5、静态网站)信息收集扫描工具

可以帮助渗透测试工程师、攻击队成员、红队成员快速收集到移动端或者静态WEB站点中关键的资产信息并提供基本的信息输出,如：Title、Domain、CDN、指纹信息、状态信息等

https://github.com/kelvinBen/AppInfoScanner

#### Next-Generation Linux Kernel Exploit Suggester

https://github.com/jondonas/linux-exploit-suggester-2

#### window redis 命令执行

可在Windows下执行系统命令的Redis模块，可用于Redis主从复制攻击。

https://github.com/0671/RedisModules-ExecuteCommand-for-Windows

#### CDK是一款为容器环境定制的渗透测试工具

CDK是一款为容器环境定制的渗透测试工具，在已攻陷的容器内部提供零依赖的常用命令及PoC/EXP。集成Docker/K8s场景特有的 逃逸、横向移动、持久化利用方式，插件化管理。

https://github.com/cdk-team/CDK

#### WordPress漏扫

https://github.com/wpscanteam/wpscan

#### 微信小程序反编译

https://github.com/xuedingmiaojun/wxappUnpacker

#### shiro支持对Shiro550（硬编码秘钥）和Shiro721（Padding Oracle）的一键化检测，支持多种回显方式

https://github.com/feihong-cs/ShiroExploit-Deprecated

#### A list of useful payloads and bypass for Web Application Security and Pentest/CTF

https://github.com/swisskyrepo/PayloadsAllTheThings

#### socks代理

https://github.com/sensepost/reGeorg

#### Android漏洞工具集

https://github.com/Juude/droidReverse

#### AI分析恶意代码

https://github.com/oasiszrz/XAIGen

## 信息收集

#### api

ceye,io.

fofa

hunter

quake

zoomeye

shodan

https://haveibeenpwned.com/

https://www.reg007.com/

https://github.com/se55i0n/Awvs_Nessus_Scanner_API

#### Gophish

Go语言编写的一款钓鱼平台

https://github.com/gophish/gophish/

#### pricking

[Pricking](https://github.com/Rvn0xsy/Pricking) 是一个自动化部署水坑和网页钓鱼的项目

https://github.com/Rvn0xsy/Pricking

## fuzz

https://github.com/AFLplusplus/AFLplusplus

## other

#### fq

https://github.com/freefq/free

https://github.com/ugvf2009/Miles

#### INSTALL openvpn

https://github.com/Nyr/openvpn-install

#### 代码分析引擎 CodeQL

https://codeql.github.com/

*<!-- more -->* 