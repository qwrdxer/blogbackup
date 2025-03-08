---
title: go-rod学习
categories:
  - 爬虫
date: 2023-10-06 11:22:48
tags:
---



https://github.com/go-rod/rod

https://pkg.go.dev/github.com/go-rod/go-rod-chinese#pkg-types



# 1. go-rod简介



## 1.1 关闭headless 进行debug

```go
package main

import (
	"github.com/go-rod/rod"
	"github.com/go-rod/rod/lib/input"
	"github.com/go-rod/rod/lib/launcher"
	"github.com/go-rod/rod/lib/utils"
	"time"
)

func main() {
	l := launcher.New().
		Headless(true)
	defer l.Cleanup()
	url_control := l.MustLaunch()
	// 创建一个浏览器实例
	browser := rod.New().
		ControlURL(url_control).
		Trace(true).
		SlowMotion(2 * time.Second).
		MustConnect()

	// 延迟关闭浏览器
	defer browser.MustClose()

	// 创建一个页面实例并访问 Bing 搜索页面
	page := browser.MustPage("https://www.bing.com").MustWaitLoad()

	// 在搜索框中输入关键词 "web"
	page.MustElement("#sb_form_q").MustInput("web").MustType(input.Enter)

	// 等待搜索结果加载完成
	page.MustWaitLoad()
    //暂停
	utils.Pause()

}

```

## 1.2 提取页面所有url

```go
	page_info, _ := page.Info()
	current_url, _ := url.Parse(page_info.URL) 
	domain := current_url.Scheme + "://" + current_url.Host //获取当前的http://xxxx /
	a_href, _ := page.ElementsX("//a")
	for _, element := range a_href {
		if element.MustAttribute("href") != nil {
			x := *element.MustAttribute("href")
			if strings.HasPrefix(x, "/") { 
				println(domain + x)
			} else {
				println(x)
			}

		}

	}
```



## 1.2 资源拦截

正常加载中有很多资源如 css 、字体、图片等是不需要的，我们可以进行拦截操作来降低资源消耗。

```go
	browser := rod.New().ControlURL(url_control).Trace(true).SlowMotion(2 * time.Second).MustConnect()
	// 延迟关闭浏览器
	defer browser.MustClose()

	//创建router
	router := browser.HijackRequests()
	defer router.MustStop()
	router.MustAdd("*", func(ctx *rod.Hijack) {
		//字体
		if ctx.Request.Type() == proto.NetworkResourceTypeFont {
			ctx.Response.Fail(proto.NetworkErrorReasonBlockedByClient)
			return
		}
		// 图片
		if ctx.Request.Type() == proto.NetworkResourceTypeImage {
			ctx.Response.Fail(proto.NetworkErrorReasonBlockedByClient)
			return
		}
		//css
		if ctx.Request.Type() == proto.NetworkResourceTypeStylesheet {
			ctx.Response.Fail(proto.NetworkErrorReasonBlockedByClient)
			return
		}
		//视频
		if ctx.Request.Type() == proto.NetworkResourceTypeMedia {
			ctx.Response.Fail(proto.NetworkErrorReasonBlockedByClient)
			return
		}
		ctx.ContinueRequest(&proto.FetchContinueRequest{})
	})
	go router.Run()
```

## 1.3  封装browser

browserManager 需要哪些东西呢?

1. Launcher ，用于对browser进行配置
2. browser ， 浏览器实例
3. Filter 过滤器，对web请求中一些不必要的资源进行过滤
4. pages [] 爬虫打开的页面都放在这里方便管理
5. lock 锁 对pages的删除、

```go
```



## 1.4 使用管道: 信号管道和数据管道

通过管道可以进行多线程的数据交互

创建10个爬虫，轮循环的获取

```go
package main

import (
	"JZCrawl/pkg/engine"
	"github.com/go-rod/rod/lib/utils"
	"sync"
)

func main() {
	myBroMannager := engine.InitBrowser(false)

	go myBroMannager.Rou.Run()
	defer myBroMannager.CloseBrowser()
	urls := []string{"https://cn.bing.com/?FORM=Z9FD1", "https://cn.bing.com/rewards/dashboard", "https://cn.bing.com/?scope=web&FORM=HDRSC1", "https://cn.bing.com/images/search?q=web&FORM=HDRSC2", "https://cn.bing.com/videos/search?q=web&FORM=HDRSC3", "https://cn.bing.com/academic/search?q=web&FORM=HDRSC4", "https://cn.bing.com/maps?q=web&FORM=HDRSC7", "https://cn.bing.com/travel/search?q=web&m=flights&FORM=FBSCOP", "https://cn.bing.com/search?q=site:developer.mozilla.org+web", "https://cn.bing.com/news/search?q=site%3awww.sohu.com&FORM=NWBCLM", "https://cn.bing.com/news/search?q=site%3astock.10jqka.com.cn&FORM=NWBCLM", "https://cn.bing.com/news/search?q=site%3aopen.163.com&FORM=NWBCLM", "https://cn.bing.com/news/search?q=site%3awww.163.com&FORM=NWBCLM", "https://cn.bing.com/dict/search?q=web&FORM=BDVSP2&qpvt=w", "https://cn.bing.com/dict/search?q=web&FORM=BDVSP2", "https://cn.bing.com/search?q=web+site%3awww.zhihu.com", "https://cn.bing.com/search?q=webs+sign+in&FORM=LGWQS1", "https://cn.bing.com/search?q=webs+army&FORM=LGWQS2", "https://cn.bing.com/search?q=google+web&FORM=LGWQS3", "https://cn.bing.com/search?q=webs+sign+in&FORM=QSRE1", "https://cn.bing.com/search?q=webs+army&FORM=QSRE2", "https://cn.bing.com/search?q=google+web&FORM=QSRE3", "https://cn.bing.com/search?q=google&FORM=QSRE4", "https://cn.bing.com/search?q=webs.com&FORM=QSRE5", "https://cn.bing.com/search?q=my+webs&FORM=QSRE6", "https://cn.bing.com/search?q=web+internet&FORM=QSRE7", "https://cn.bing.com/search?q=webs+free&FORM=QSRE8", "https://cn.bing.com/search?q=web&form=P4041&sp=-1&lq=0&pq=&sc=0-0&qs=n&sk=&cvid=B3F8FBFCE9B0454494EC9C3CBCCD1291&ghsh=0&ghacc=0&ghpl=&ubiroff=1"}

	urlQuery := make(chan string, 5) //送入通道中

	// 启动10个 goroutine
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1) // 增加等待组的计数

		go CrawlUrl(myBroMannager, urlQuery)
	}
	for _, target := range urls {
		println(target)
		urlQuery <- target
	}
	// 等待所有 goroutine 完成
	wg.Wait()
	//utils.Pause()
}
func CrawlUrl(bro *engine.BrowserManager, urlQuery <-chan string) {
	for {
		select {
		case url, _ := <-urlQuery:
			page := bro.NewPage(url)
			page.MustWaitLoad()
			pageInfo, _ := page.Info()
			utils.Sleep(10)
			print(pageInfo.Title)
			page.Close()
		}
	}
	print("启动任务")

	//page.Close()
}

```

Argo 中通过管道管理了爬虫的创建 tab.go 的TabWork部分。

创建一个管道 10个容量，每一个爬虫都会占用一个位置，当爬虫结束后再次释放，而TabWork部分负责管理爬虫的产生。





## 1.5关闭时间太长  or 空白的页面

获取浏览器当前的所有页面



## 1.6 爬虫中是否要对提取到的URL进行多种处理？

高速 or 高效 的问题

如果爬虫中有很多对数据的处理，则延迟了下一个爬虫的产生，个人认为应该优先爬取，让专门的线程对爬虫结果进行处理。



---

文章参考:

博客地址: qwrdxer.github.io

欢迎交流: qq1944270374