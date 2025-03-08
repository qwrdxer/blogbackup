---
title: RustScan源代码分析
categories:
  - 代码分析
date: 2023-10-05 19:28:59
tags:
---

文章整体目录:

1. 介绍RustScan
2. 代码整体分析
3. 主函数部分
4. scanner部分



# 1.RustScan介绍



## 1.1 RustScan简介

RustScan 是基于**Rust开发的一款端口扫描器**

1. 快速**:** 得益于**Rust的并发和性能，RustScan的端口扫描速度飞快**
2. 跨平台**:** 单文件编译、部署
3. 可扩展**:内置脚本，可对接nmap**

## 1.2 项目地址

https://github.com/RustScan/RustScan



## 1.3快速使用

![fast](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/fast.gif)

# 2.代码整体分析

![整体架构](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231005193612526.png)

代码整体框架如上图所示

1. `input.rs` 用于解析用户输入的参数,如IP地址、端口范围 等。
2. `main.rs` 是代码的主逻辑。
3. `tui.rs` 是rust的一些美化终端输出操作。
4. `bentchmark/mod.rs` 用于测试RustScan的性能,这个没啥好讲的。
5. `port_strategy/*`，端口策略，我们输入的端口可能是` 22,25,80,443`这种形式,也可能是 `1-1000` 这种范围形式，` port_strategy ` 提供代码将其转换成rust的数组形式,如可以将`"1-1000"` 转换成 `[1,2,3,4 ... 1000]`。
6. `Scanner/*` , 扫描主逻辑,通过创建sokcet来验证端口是否开放。
7. ` script/mod.rs` 脚本模块

 代码中最为重要的部分是主函数 `main.rs` , 扫描器 `Scanner `  , 脚本引擎 `script`，本文主要分析` main` 和`scanner`部分的代码同时尝试编写一个脚本。







# 3.主函数部分

我们调用rustscan的命令如下 `rustscan.exe -a 172.22.105.149,172.22.105.148 -b 5000 -r 1-10000`

- -a 参数指定了要扫描的ip地址,多个地址按 逗号分隔
- -b  batch_size 指定要同时扫描多少个端口
- -r  port_range 指定扫描的端口  支持逗号间隔的具体端口形式 或者 - 间隔的范围形式



## 3.1 主函数流程图

![main.rs的主流程](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231005194520965.png)

如上图所示, 主函数的调用流程可以分为三部分。

1. 首先是初始化部分,首先将命令行的参数进行解析,随后对脚本进行预加载,我们的命令没有显示的指定脚本,RustScan会默认调用`nmap`对扫描结果进一步处理
2. 扫描部分,这是RustScan的主要代码部分,通过初始化部分得到的参数来构造扫描器执行端口扫描
3. 后续处理部分, 扫描的结果是socket(ip,port) 形式,rust要将其聚合成更方便后续处理的形式并调用脚本(如果有的话)进行最终处理。



## 3.2 main.rs代码分析

`main.rs`的代码有约340行, 主函数 `main ()` 有150行, 还有 `parse_addresses` `parse_address` 等函数,主要是将字符串形式的 ip转换成 ipaddr的形式,如 "172.15.22.11" -> IpAddr(172.15.22.11) 。

总之我们重点关注主函数的代码即可，其他部分感兴趣的可以自行阅读一下。


①初始化部分( 55~81行)

> ![初始化部分](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231005195855197.png)
>
> 
>
> ```rust
>     env_logger::init();
>     let mut benchmarks = Benchmark::init();
>     let mut rustscan_bench = NamedTimer::start("RustScan");
> ```
>
> 这三行对主逻辑没啥影响,可以大概知道 env_logger::init()启动了日志模块,其后面的两行是启动了bentchmark的性能测试计时。
>
> 
>
> ```rust
> let mut opts: Opts = Opts::read();
> let config = Config::read(opts.config_path.clone());
> opts.merge(&config);
> ```
>
> Opts结构体存储在`input.rs`中,  通过调用` Opts::read()` 可以将用户输入的命令行参数存储到 opts中。
>
> 同时如果在指定目录下(用户的home目录)有`.rustscan.toml` 这个配置文件,RustScan也会读取这个配置文件中存储的参数并合并到`opts`变量中。
>
> 
>
> 若我们输入的命令为``rustscan.exe -a 172.22.105.149,172.22.105.148 -b 5000 -r 1-10000` ，则`opts` 变量最终结果如下:
>
> ![opts存储的信息](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231005200256593.png)
>
> 
>
> 
>
> 加载完命令行参数后就是脚本的预加载,通过调用` init_scripts` 来加载脚本,详细的代码分析在后续部分,这里我们命令行中没有指定脚本因此会使用默认的nmap脚本
>
> ![调用init_scripts](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231005200749393.png)
>
> ![image-20231005200937210](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231005200937210.png)
>
> 



②处理IP ，将其转换成IpAddr形式

>  现在所有信息都存储在`opts`中,但我们还是无法直接使用这些参数构造扫描器, 如ip地址还是字符串形式, 我们首先要将其转换成IpAddr的格式
>
> ![调用 parse_addresses来处理IP地址](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231005201258494.png)
>
> 具体的处理代码在`main.rs`的后半部分, 感兴趣的可以自己看看,总之我们最终获得了一个变量`ips`存储了所有目标IP地址。
>
> ![image-20231005201511862](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231005201511862.png)
>
> 
>
> 端口处理的部分在后续的`Scanner`模块的`run`方法中。





③ scanner的构造运行

> ![构造runner](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231005202427874.png)
>
>
> ```rust 
> const AVERAGE_BATCH_SIZE: u16 = 3000; //在main.rs的一开始设置的
> 
> let batch_size: u16 = AVERAGE_BATCH_SIZE; //常量3000
> ```
>
> 写到这里才发现，scanner指定batch_size 并不是从opts取出的,默认是设置为3000了,这应该是RustScan多次测试得出的一个比较合适的批量大小,当然要是想自定义batchsize只要稍微修改一下就行，我们后续就按batch_size=3000好了
>
> ```rust
> batch_size -> opts.batch_size
> ```
>
> 通过`Scanner::new()`来构造一个scanner,其具体的值如下:
>
> ![构造好的 scanner](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231005202501535.png)
>
> 
>
> ```rust
> let scan_result = block_on(scanner.run());
> ```
>
> 构造好scanner后，可以开始进行扫描了, `scanner::run()`会执行这个任务直到扫描完全部的端口。因为是异步执行的代码 ,使用block_on 来等待执行结果，随后将结果返回给`scan_result` , 其每一个成员都是一个socketAddr，按IP:port的格式记录了开放的端口。
>
> ![扫描结果](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231005204202175.png)



④后续处理

> 现在scan_result存储的是IP:PORT对,RustScan 对其进行了进一步的处理 ,处理成的格式为 IP:[port1,port2 ...]
>
> ![后续处理](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231005205125568.png)
>
> ```rust
>     let mut ports_per_ip = HashMap::new();
> ```
>
> 创建一个hashmap , 键为IP 值为port数组。
>
> ```rust
>     for socket in scan_result {
>         ports_per_ip
>             .entry(socket.ip())
>             .or_insert_with(Vec::new)
>             .push(socket.port());
>     }
> ```
>
> 遍历 scan_result的结果,将端口都聚集到指定的IP上。
>
> 后面的`for ip in ips ...` 就是将没有开放端口的IP打印出来。
>
> 
>
> 最终` ports_per_ip`  结果如下, first部分存储的是IP , second部分存储的是对应开放的端口。
>
> ![数据格式](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231005204535805.png)

后续就是调用脚本对其进行处理，在script部分进行更详细的分析。

以上就是主函数的逻辑，接下来让我们详细分析分析scanner部分和script部分。









# 4.scanner部分



## 4.1 Scanner模块整体分析

![image-20231005205735514](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231005205735514.png)

首先mod.rs是主体代码, 它包含了Scanner的创建、任务执行等代码

Socket_iterator.rs是用来生成socket的，比如有2个目标IP，要扫描1000个IP ，则一共生成2000个socket(IP:PORT) 。

 

## 4.2 代码分析

①Scanner结构体

> ![scanner结构体](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231007205227084.png)
>
> 我们在主函数中构建Scanner时，通过opts 将目标IP、扫描端口、批量大小等信息对应的参数传入即可 



②通过new 方法创建 Scanner

> ![new创建结构体](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231007205626995.png)
>
> 主函数调用的就是这个new函数来创建的Scanner 
>
> ![Scanner的debug](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231007205715493.png)
>
> 通过debug可知 Scanner结构体,包含扫描的ip地址，扫描的端口、超时时间、尝试次数等
>
> ![Scanner结构体](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231007205824052.png)

③调用scanner调用 run方法执行端口扫描

> ![run方法](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231011200922766.png)
>
> 首先将scanner的端口信息部分转换成`[1,2,3....1000]` 这种集合
>
> 然后调用`ScoketIterator` 创建一个 迭代器，迭代器每次返回一个Socket(IP:PORT )
>
> ![迭代器返回值示例](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231011201144776.png)
>
> 随后创建一个变量`open_socket` 来存储后续扫描中发现的开放的端口， 创建`ftrs` 用于执行异步的端口扫描 。
>
> 
>
> 首先创建`batch_size` 个端口扫描任务,代码如下
>
> ![run方法-2](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231011201504340.png)
>
> `scan_socket` 函数会用参数中的Socket来尝试建立连接，如果成功连接则代表目标端口是开放的。
>
> 接下来的思路就是：等待`ftrs`中的任务完成，每完成一个新的任务都会立刻往`ftrs`中填入新的任务，确保同时有`batch_size`个任务在执行。
>
> ![run方法3](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231011202109732.png)
> 
>
> ` while let Some(result) = ftrs.next().await` 代码会阻塞到有任务完成，然后进入到while代码中，首先若是`socket_iterator`中还有待执行的扫描任务，则将其加入`ftrs`中， 后续对`result`进行分析，若是返回ok则代表端口开放，将其放到`open_sockets`变量中。
>
> ![Scanner中的变量存储](RustScan%E6%BA%90%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90/image-20231011202452589.png)
>
> 总之，扫描完成后，开放的端口会以socket(ip:port)的形式存储到scan_result变量中 ，主函数会对其进行后续的处理
>
> 
> 









---



博客地址: qwrdxer.github.io

欢迎交流: qq1944270374