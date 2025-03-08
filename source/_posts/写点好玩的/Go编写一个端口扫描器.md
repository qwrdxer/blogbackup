---
title: Go编写一个扫描器
toc: true
categories:
  - 写点好玩的
date: 2022-04-10 14:34:18
tags:
---

## 基础学习

### 端口连接建立



### 版本1 简单的扫描

```go
// panic_recover.go
package main

import (
	"fmt"
	"net"
)

func Scan(IP string) []int {
	var target_tmplet = "%s:%d"
	var index_open = 0
	open_port := make([]int, 5, 5)
	scan_port := []int{22, 80, 6379}
	for _, port := range scan_port {
		address := fmt.Sprintf(target_tmplet, IP, port)
		conn, err := net.Dial("tcp", address)
		if err != nil {
			fmt.Println(err)
			continue
		}
		conn.Close()
		fmt.Printf("%d port open!\n", port)
		open_port[index_open] = port
		index_open++
	}
	return open_port
}
func main() {
	Scan("scanme.nmap.org")
}

```



*<!-- more -->* 

### 版本2，结构体封装

将代码进行封装，方便操作及功能拓展

```
// panic_recover.go
package main

import (
	"fmt"
	"net"
)

type PortScanTask struct {
	target        string   //扫描目标, 可以是IP 或 域名
	scanPortRange []string //存储扫描的范围，可以是具体的端口(如80,8080,9090)，也可以是某个范围(如1-100)
	openPort      []string // 用于存储开放端口
	openPortCount int      // 记录开放端口数量，在存储开放端口时，作为下标存储
}

/*
	用于初始化端口扫描任务,主要是对用户输入的端口范围进行处理
	如用户输入为 80,8080,3306,1-100
	我们首先需要按照逗号进行切分，然后检查这些输入是否符合规范即可
*/
func (p *PortScanTask) init(target string, targetRanget string) {
	p.openPortCount = 0
	p.target = target
	p.scanPortRange = []string{"22", "80", "6379", "3306"}
	p.openPort = make([]string, 5, 5)
}
func (p *PortScanTask) Scan() []string {
	var target_tmplet = "%s:%s"
	for _, port := range p.scanPortRange {
		address := fmt.Sprintf(target_tmplet, p.target, port)
		conn, err := net.Dial("tcp", address)
		if err != nil {
			fmt.Println(err)
			continue
		}
		conn.Close()
		fmt.Printf("%s port open!\n", port)
		p.openPort[p.openPortCount] = port
		p.openPortCount++
	}
	return p.openPort
}
func main() {
	var task1 PortScanTask
	task1.init("121.5.73.12", "test")
	task1.Scan()
}

```

暂时还未实现输入端口切割功能！， 先把多线程搞上



### 版本3: 多线程



```go
// panic_recover.go
package main

import (
	"fmt"
	"net"
	"sort"
)

type PortScanTask struct {
	target        string   //扫描目标, 可以是IP 或 域名
	target_tmplet string   // 连接的模板 值为 %s:%s 在 init函数中进行赋值
	scanPortRange []string //存储扫描的范围，可以是具体的端口(如80,8080,9090)，也可以是某个范围(如1-100)
	openPort      []string // 用于存储开放端口
	openPortCount int      // 记录开放端口数量，在存储开放端口时，作为下标存储
	threads       int      //多线程数量
}

/*
	用于初始化端口扫描任务,主要是对用户输入的端口范围进行处理
	如用户输入为 80,8080,3306,1-100
	我们首先需要按照逗号进行切分，然后检查这些输入是否符合规范即可
*/
func (p *PortScanTask) init(target string, targetRanget string) {
	p.openPortCount = 0
	p.target = target
	p.target_tmplet = "%s:%s"
	p.scanPortRange = []string{"22", "80", "6379", "3306", "111", "3333", "53", "7777"}
	p.openPort = make([]string, 2)
	p.threads = 5
}

/*
	我们创建一个worker 方法，他从ports通道中取出端口进行扫描，然后将结果返回给results通道
*/
func (p *PortScanTask) PortScanWorker(ports, results chan string) {
	for port := range ports {
		address := fmt.Sprintf(p.target_tmplet, p.target, port)
		conn, err := net.Dial("tcp", address)
		if err != nil {
			//如果端口没有开放,则向通道中写入0
			results <- "0"
			continue
		}
		conn.Close()
		results <- port
	}
}
func (p *PortScanTask) Scan() []string {
	portchan := make(chan string, p.threads)
	resultchan := make(chan string)
	for i := 0; i < p.threads; i++ { //创建 5个worker
		go p.PortScanWorker(portchan, resultchan)
	}
	go func() { //启动一个线程将端口写入通道中
		for _, port := range p.scanPortRange {
			portchan <- port
		}
	}()

	for i := 0; i < len(p.scanPortRange); i++ {
		port := <-resultchan
		if port != "0" {
			p.openPort = append(p.openPort, port)
		}
	}
	close(portchan)
	close(resultchan)
	sort.Strings(p.openPort)
	return p.openPort
}
func main() {
	var task1 PortScanTask
	task1.init("121.5.73.12", "111,22,3306,53,80")
	t := task1.Scan()
	fmt.Println(t)
}

```



### 计算端口数量

```go
// panic_recover.go
package main

import (
	"fmt"
	"net"
	"sort"
	"strconv"
	"strings"
)

type PortScanTask struct {
	target        string //扫描目标, 可以是IP 或 域名
	target_tmplet string // 连接的模板 值为 %s:%s 在 init函数中进行赋值
	scanPortRange []int  //存储扫描的范围，可以是具体的端口(如80,8080,9090)，也可以是某个范围(如1-100)
	openPort      []int  // 用于存储开放端口
	openPortCount int    // 记录开放端口数量，在存储开放端口时，作为下标存储
	threads       int    //多线程数量
}

/*
	用于初始化端口扫描任务,主要是对用户输入的端口范围进行处理
	如用户输入为 80,8080,3306,1-100
	我们首先需要按照逗号进行切分，然后检查这些输入是否符合规范即可
*/
func (p *PortScanTask) init(target string, targetRanget string) {
	p.openPortCount = 0
	p.target = target
	p.target_tmplet = "%s:%d"
	p.scanPortRange = make([]int, 0)
	p.openPort = make([]int, 0)
	p.threads = 100

	p.VertifyParm(targetRanget)
}

/*
	我们创建一个worker 方法，他从ports通道中取出端口进行扫描，然后将结果返回给results通道
*/
func (p *PortScanTask) PortScanWorker(ports chan int, results chan int) {
	for port := range ports {
		address := fmt.Sprintf(p.target_tmplet, p.target, port)
		conn, err := net.Dial("tcp", address)
		if err != nil {
			//如果端口没有开放,则向通道中写入0
			results <- 0
			//fmt.Printf("%d not open!\n", port)
			continue
		}
		conn.Close()
		fmt.Printf("%d open!\n", port)
		results <- port
	}
}
func (p *PortScanTask) Scan() []int {
	portchan := make(chan int, p.threads)
	resultchan := make(chan int)
	for i := 0; i < p.threads; i++ { //创建 5个worker
		go p.PortScanWorker(portchan, resultchan)
	}
	go func() { //启动一个线程将端口写入通道中
		for _, port := range p.scanPortRange {
			portchan <- port
		}
	}()

	for i := 0; i < len(p.scanPortRange); i++ {
		port := <-resultchan
		if port != 0 {
			p.openPort = append(p.openPort, port)
		}
	}
	close(portchan)
	close(resultchan)
	sort.Ints(p.openPort)
	return p.openPort
}

//验证端口输入的参数是否符合规范, 一串合规的的输入如下: 22,80,8080,1000-3999
func (p *PortScanTask) VertifyParm(input string) {
	sliceParm := strings.Split(input, ",")
	for i := range sliceParm {
		if (strings.Index(sliceParm[i], "-")) == -1 { //如果不是1-100这种范围的类型
			tmpInt, err := strconv.Atoi(sliceParm[i]) //转换成数字
			if err != nil {
				fmt.Printf("%s参数不正确!", sliceParm[i])
			}
			if tmpInt <= 0 || tmpInt > 65535 {
				fmt.Printf("%s参数不正确!", sliceParm[i])
			}
			p.scanPortRange = append(p.scanPortRange, tmpInt)
			fmt.Println(tmpInt)
		} else { //如果是1-100这种范围的类型
			subsliceParm := strings.Split(sliceParm[i], "-")
			if len(subsliceParm) != 2 {
				fmt.Printf("%s参数不正确!", sliceParm[i])
			} else {
				if subsliceParm[0] == "" || subsliceParm[1] == "" {
					fmt.Printf("%s参数不正确!", sliceParm[i])
				} else { //到此为止才能判断是正确的num1-num2类型
					tmpInt1, err1 := strconv.Atoi(subsliceParm[0])
					tmpInt2, err2 := strconv.Atoi(subsliceParm[1])
					if err1 != nil || err2 != nil {
						fmt.Printf("参数不正确")
					}
					if tmpInt2-tmpInt1 <= 0 || tmpInt2 > 65535 || tmpInt1 > 65535 {
						fmt.Printf("参数不正确")
					}
					for i := tmpInt1; i <= tmpInt2; i++ {
						p.scanPortRange = append(p.scanPortRange, i)
					}
				}
			}
		}

	}

}
func main() {
	var task1 PortScanTask
	task1.init("121.5.73.12", "1-65535")
	t := task1.Scan()
	fmt.Println(t)
}

```



### 加入参数解析

```go
// panic_recover.go
package main

import (
	"flag"
	"fmt"
	"net"
	"sort"
	"strconv"
	"strings"
	"time"
)

type PortScanTask struct {
	target        string //扫描目标, 可以是IP 或 域名
	target_tmplet string // 连接的模板 值为 %s:%s 在 init函数中进行赋值
	scanPortRange []int  //存储扫描的范围，可以是具体的端口(如80,8080,9090)，也可以是某个范围(如1-100)
	openPort      []int  // 用于存储开放端口
	openPortCount int    // 记录开放端口数量，在存储开放端口时，作为下标存储
	Workers       int    //多线程数量
}

/*
	用于初始化端口扫描任务,主要是对用户输入的端口范围进行处理
	如用户输入为 80,8080,3306,1-100
	我们首先需要按照逗号进行切分，然后检查这些输入是否符合规范即可
*/
func (p *PortScanTask) init(target string, targetRanget string) {
	p.openPortCount = 0
	p.target = target
	p.target_tmplet = "%s:%d"
	p.scanPortRange = make([]int, 0)
	p.openPort = make([]int, 0)
	p.Workers = 3000

	p.VertifyParm(targetRanget)
}

/*
	我们创建一个worker 方法，他从ports通道中取出端口进行扫描，然后将结果返回给results通道
*/
func (p *PortScanTask) PortScanWorker(ports chan int, results chan int) {
	for port := range ports {
		address := fmt.Sprintf(p.target_tmplet, p.target, port)
		conn, err := net.Dial("tcp", address)
		if err != nil {
			//如果端口没有开放,则向通道中写入0
			results <- 0
			//fmt.Printf("%d not open!\n", port)
			continue
		}
		conn.Close()
		fmt.Printf("%d open!\n", port)
		results <- port
	}
}
func (p *PortScanTask) Scan() []int {
	portchan := make(chan int, 1000)
	resultchan := make(chan int)
	for i := 0; i < p.Workers; i++ { //创建 5个worker
		go p.PortScanWorker(portchan, resultchan)
	}
	go func() { //启动一个线程将端口写入通道中
		for _, port := range p.scanPortRange {
			portchan <- port
		}
	}()

	for i := 0; i < len(p.scanPortRange); i++ {
		port := <-resultchan
		if port != 0 {
			p.openPort = append(p.openPort, port)
		}
	}
	close(portchan)
	close(resultchan)
	sort.Ints(p.openPort)
	return p.openPort
}

//验证端口输入的参数是否符合规范, 一串合规的的输入如下: 22,80,8080,1000-3999
func (p *PortScanTask) VertifyParm(input string) {
	sliceParm := strings.Split(input, ",")
	for i := range sliceParm {
		if (strings.Index(sliceParm[i], "-")) == -1 { //如果不是1-100这种范围的类型
			tmpInt, err := strconv.Atoi(sliceParm[i]) //转换成数字
			if err != nil {
				fmt.Printf("%s参数不正确!", sliceParm[i])
			}
			if tmpInt <= 0 || tmpInt > 65535 {
				fmt.Printf("%s参数不正确!", sliceParm[i])
			}
			p.scanPortRange = append(p.scanPortRange, tmpInt)
		} else { //如果是1-100这种范围的类型
			subsliceParm := strings.Split(sliceParm[i], "-")
			if len(subsliceParm) != 2 {
				fmt.Printf("%s参数不正确!", sliceParm[i])
			} else {
				if subsliceParm[0] == "" || subsliceParm[1] == "" {
					fmt.Printf("%s参数不正确!", sliceParm[i])
				} else { //到此为止才能判断是正确的num1-num2类型
					tmpInt1, err1 := strconv.Atoi(subsliceParm[0])
					tmpInt2, err2 := strconv.Atoi(subsliceParm[1])
					if err1 != nil || err2 != nil {
						fmt.Printf("参数不正确")
					}
					if tmpInt2-tmpInt1 <= 0 || tmpInt2 > 65535 || tmpInt1 > 65535 {
						fmt.Printf("参数不正确")
					}
					for i := tmpInt1; i <= tmpInt2; i++ {
						p.scanPortRange = append(p.scanPortRange, i)
					}
				}
			}
		}

	}

}
func main() {

	IPString := flag.String("IP", "", "要扫描的目标IP")

	PORTString := flag.String("PORT", "", "要扫描的端口")

	flag.Parse()
	start := time.Now() // 获取当前时间
	var task1 PortScanTask
	task1.init(*IPString, *PORTString)
	task1.Scan()
	cost := time.Since(start)
	fmt.Println("cost:", cost)
}

```



*<!-- more -->* 







> 参考文章
>
> 