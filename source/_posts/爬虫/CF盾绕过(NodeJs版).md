---
title: CF盾绕过(NodeJs版)
categories:
  - 爬虫
date: 2023-12-22 20:11:48
tags:
---





## 前期准备

https://yescaptcha.com/

用于CF盾绕过





https://www.stormproxies.cn/api


![image-20231222201258922](CF%E7%9B%BE%E7%BB%95%E8%BF%87(NodeJs%E7%89%88)/image-20231222201258922.png)

这个代理用于转发流量



## 实现思路



使用Yescaptcha 通过代理访问CF盾 ，Yescaptcha 绕过后返回cookie， 主机通过这个cookie借助代理访问即可完成绕过。



## 代码部分

### 一、 Yescaptcha 绕过 CF 5秒盾



首先是任务创建、轮训任务状态、 获取结果的代码

```js
async function createTask(url, proxy) {
  const data = {
    // Fill in your own client key
    "clientKey": clientKey,
    "task": {
      "type": "CloudFlareTaskS1",
      "websiteURL": url,
      "proxy": proxy,
    },
  };
  const apiUrl = "https://api.yescaptcha.com/createTask";

  try {
    const response = await axios.post(apiUrl, data);
    return response.data;
  } catch (error) {
    console.error(
      "Error creating task:",
      error.response ? error.response.data : error.message
    );
    throw error;
  }
}
//获取任务状态
async function getTask(taskId) {
  const apiUrl = "http://api.yescaptcha.com/getTaskResult";
  const data = {
    // Fill in your own client key
    clientKey: clientKey,
    taskId: taskId,
  };

  try {
    const response = await axios.post(apiUrl, data);
    return response.data;
  } catch (error) {
    console.error(
      "Error getting task result:",
      error.response ? error.response.data : error.message
    );
    throw error;
  }
}
async function getResult(url, proxy) {
  const uuid = await createTask(url, proxy);

  if (!uuid || !uuid.taskId) {
    return uuid;
  }

  console.log("TaskID:", uuid.taskId);

  for (let i = 0; i < 30; i++) {
    await delay(3000);
    const result = await getTask(uuid.taskId);

    if (result.status === "processing") {
      continue;
    } else if (result.status === "ready") {
      return result;
    } else {
      console.log("Fail:", result);
      return result; // You might want to handle failure differently
    }
  }
}

```



我们可以通过如下代码来绕过CF盾

```js
  const task = await getResult("https://faucetgreenlist.snarkos.net", proxy);
  const cookies=task.solution.cookies;
  const headers =task.solution.headers;
```



### 二、浏览器设置cookie,绕过CF 5秒盾

首先需要将cookie转换成 puppyter能接受的形式

```bash
{
  name: '__cf_bm',
  value: 'w8UDIQ4eETjmAQziMCwRmvyfzAzotHW.Py9wCpa3T0E-1703251091-1-AcJq72upiNiqlVNosTebMiSad2GAvOW/kkGT3zt8NkNZCVRiinVAY1mdXCw7JR3gHt4QKY6q+B3iPdWTnFvpzgM=',
  domain: '.snarkos.net',
  path: '/',
  expires: 1703252891.215921,
  size: 152,
  httpOnly: true,
  secure: true,
  session: false,
  sameSite: 'None',
  sameParty: false,
  sourceScheme: 'Secure',
  sourcePort: 443
}
```

思路： 访问网站，获取到cookie的模板，将Yescaptcha破解得到的Cookie 转换成puppeteer可以识别的形式

```javascript
async function convCookie(target, templete_cookie) {
  let newCookies = [];
  for (const [key, value] of Object.entries(target)) {
    templete_cookie.name = key;
    templete_cookie.value = value;
    newCookies.push({ ...templete_cookie })

  }
  return newCookies;
}
```





### 三、使用YesCaptcha 绕过第二层CF盾



![image-20231227203348967](CF%E7%9B%BE%E7%BB%95%E8%BF%87(NodeJs%E7%89%88)/image-20231227203348967.png)

绕过五秒盾后，第二个为点击验证 ，需要使用YesCaptcha绕过

https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/pages/61734913/TurnstileTaskProxyless+CloudflareTurnstile

思路:  发送websiteKey 到YesCaptcha ,  将返回结果中的数据设置到浏览器中

<input type="hidden" name="cf-turnstile-response" id="cf-chl-widget-0ua84_response" value="">



代码如下:

````javascript
async function createTurnstileTask(url, proxy, websiteKey) {
  const data = {
    "clientKey": clientKey,
    "task":
    {
      "type": "TurnstileTaskProxyless",
      "websiteURL": url,
      "websiteKey": websiteKey,
      "proxy": proxy,
    }
  }


  try {
    const response = await axios.post(apiUrl, data);
    return response.data;
  } catch (error) {
    console.error(
      "Error creating task:",
      error.response ? error.response.data : error.message
    );
    throw error;
  }
}

async function getTurnstileResult(url, proxy, websiteKey) {
  const uuid = await createTurnstileTask(url, proxy, websiteKey);

  if (!uuid || !uuid.taskId) {
    return uuid;
  }

  console.log("TaskID:", uuid.taskId);

  for (let i = 0; i < 30; i++) {
    await delay(3000);
    const result = await getTask(uuid.taskId);

    if (result.status === "processing") {
      continue;
    } else if (result.status === "ready") {
      return result;
    } else {
      console.log("Fail:", result);
      return result; // You might want to handle failure differently
    }
  }
}

````



全部代码

````javascript
const puppeteer = require("puppeteer");
const axios = require("axios");
//E3{i7JHs)_v3F}@H
// 代理网络: eu.stormip.cn
// 端口: 1000
// 账户名: storm-qwrdxer_area-US_session-123456_life-5
// 密码: qwrdxer
// Yes的key
const clientKey = "1adad62a440476bdcf6526fc98dee236842900cd30644";
const apiUrl = "https://api.yescaptcha.com/createTask";
proxy =
  "http://storm-qwrdxer_area-US_session-123456_life-5:qwrdxer@eu.stormip.cn:1000";
proxies = {
  http: proxy,
  https: proxy,
};
//要访问的网址
const delay = (milliseconds) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));

//创建任务 CloudFlare5
async function createCloudFlare5Task(url, proxy) {
  const data = {
    // Fill in your own client key
    "clientKey": clientKey,
    "task": {
      "type": "CloudFlareTaskS1",
      "websiteURL": url,
      "proxy": proxy,
    },
  };


  try {
    const response = await axios.post(apiUrl, data);
    return response.data;
  } catch (error) {
    console.error(
      "Error creating task:",
      error.response ? error.response.data : error.message
    );
    throw error;
  }
}
//获取任务状态
async function getTask(taskId) {
  const apiUrl = "http://api.yescaptcha.com/getTaskResult";
  const data = {
    // Fill in your own client key
    clientKey: clientKey,
    taskId: taskId,
  };

  try {
    const response = await axios.post(apiUrl, data);
    return response.data;
  } catch (error) {
    console.error(
      "Error getting task result:",
      error.response ? error.response.data : error.message
    );
    throw error;
  }
}
async function getResult(url, proxy) {
  const uuid = await createCloudFlare5Task(url, proxy);

  if (!uuid || !uuid.taskId) {
    return uuid;
  }

  console.log("TaskID:", uuid.taskId);

  for (let i = 0; i < 30; i++) {
    await delay(3000);
    const result = await getTask(uuid.taskId);

    if (result.status === "processing") {
      continue;
    } else if (result.status === "ready") {
      return result;
    } else {
      console.log("Fail:", result);
      return result; // You might want to handle failure differently
    }
  }
}
async function createTurnstileTask(url, proxy, websiteKey) {
  const data = {
    "clientKey": clientKey,
    "task":
    {
      "type": "TurnstileTaskProxyless",
      "websiteURL": url,
      "websiteKey": websiteKey,
      "proxy": proxy,
    }
  }


  try {
    const response = await axios.post(apiUrl, data);
    return response.data;
  } catch (error) {
    console.error(
      "Error creating task:",
      error.response ? error.response.data : error.message
    );
    throw error;
  }
}

async function getTurnstileResult(url, proxy, websiteKey) {
  const uuid = await createTurnstileTask(url, proxy, websiteKey);

  if (!uuid || !uuid.taskId) {
    return uuid;
  }

  console.log("TaskID:", uuid.taskId);

  for (let i = 0; i < 30; i++) {
    await delay(3000);
    const result = await getTask(uuid.taskId);

    if (result.status === "processing") {
      continue;
    } else if (result.status === "ready") {
      return result;
    } else {
      console.log("Fail:", result);
      return result; // You might want to handle failure differently
    }
  }
}
//curl --proxy brd.superproxy.io:22225 --proxy-user brd-customer-hl_11ca8dd0-zone-zone1:ihqk884qno7g  https://lumtest.com/myip.json

//将原始的cookies 转换成puttpyter 识别的Cookie的 形式
async function convCookie(target, templete_cookie) {
  let newCookies = [];
  for (const [key, value] of Object.entries(target)) {
    templete_cookie.name = key;
    templete_cookie.value = value;
    newCookies.push({ ...templete_cookie })

  }
  return newCookies;
}
(async () => {

  //打开浏览器 启用插件(puppeteer默认禁用)
  const browser = await puppeteer.launch({
    headless: "new",
    args: [
      //      `--disable-extensions-except=${StayFocusd}`,
      //      `--load-extension=${StayFocusd}`,
      "--enable-automation",
      //国内需要代理
      "--proxy-server=eu.stormip.cn:1000",
      '--no-sandbox'
    ],
  });

  const page = await browser.newPage();
  page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36');
  await page.authenticate(
    { username: 'storm-qwrdxer_area-US_session-123456_life-5', password: 'qwrdxer' }
  )
  await page.goto("https://faucetgreenlist.snarkos.net");
  //获取一个cookie作为模板
  let target_cookies = await page.cookies();
  //模板cookie
  let templete_cookie = target_cookies[0];
  console.log(templete_cookie)
  //用到的参数为: 
  //绕过cloudflare
  const task = await getResult("https://faucetgreenlist.snarkos.net", proxy);
  //绕过成功，浏览器继承这个session
  let tt = await convCookie(task.solution.cookies, templete_cookie);
  await page.setCookie(...tt);
  await page.setUserAgent(task.solution.user_agent)
  await page.goto("https://faucetgreenlist.snarkos.net");
  //首先,访问页面获取到cookie的模板
  // const text = await page.content();
  // console.log(text)
  await page.screenshot({ path: 'screenshot.png' });
  //await page.setExtraHTTPHeaders(headers)
  const turnstileResult = await getTurnstileResult("https://faucetgreenlist.snarkos.net/", proxy, "0x4AAAAAAALpOsoiFJlMjdUA");
  console.log(turnstileResult.solution.token)
  // const frame = await page
  //   .frames()
  //   .find((frame) =>
  //     frame.url().includes("https://challenges.cloudflare.com/cdn-cgi/challenge-platform")
  //   );

  // const frametext = await frame.content()
  // console.log(frametext)

  await page.evaluate((token) => {
    const inputElement = document.querySelector('[name="cf-turnstile-response"]');
    if (inputElement) {
      inputElement.value = token;
    }
  }, turnstileResult.solution.token);
  console.log("1111111111111111111111111111111")
  await delay(3000)
  // const text2 = await page.content();
  // console.log(text2)
  await page.screenshot({ path: 'screenshot_2.png' });
  await page.type('#user_input', 'aleo1t3e485lj323rwl77dhtfzyw7tjz4qngxpj7yg5jjvydvxy6g4sxqztuyxf');
  await page.click('input[value="Paint it green!"]')
  await delay(3000)
  await page.screenshot({ path: 'screenshot_3.png' });
})();


````













## 代码整合入bot中























---

文章参考:

博客地址: qwrdxer.github.io

欢迎交流: qq1944270374