---
title: MCP(Model Context Protocol) 简介
date: 2025-03-12 15:47:27
categories:
- MCP
tags:
- MCP
- 通讯协议
- LLM
---



###  资料

开发文档
https://modelcontextprotocol.io/introduction

claude desktop app下载
https://claude.ai/download

规范
https://spec.modelcontextprotocol.io/



资源
https://github.com/punkpeye/awesome-mcp-servers 例子
https://modelcontextprotocol.io/examples   官方例子



### 概述

#### 原始的LLM
只根据用户输入进行输出, 缺点是无法与外界交互

> User: 1+1=?
> Assistant: 2

> User:今天上海天气如何
> Assistant:您好，建议您联网获取时效性较强的信息；如果还有其他问题需要帮助，请随时告诉我！

#### Function call
用户提供一系列可用的函数, 大模型根据用户输入**选择**合适的函数，给出具体参数返回给用户.
用户调用函数并将结果发送给大模型，大模型拥有了外界的信息
https://www.youtube.com/watch?v=Qor2VZoBib0

![image-20250312154940796](MCP(Model%20Context%20Protocol)%20%E7%AE%80%E4%BB%8B/image-20250312154940796.png)
**Key Point**

- 应用将用户的问题、可使用的工具发送给LLM
- LLM将要调用的**函数**和**参数**返回
- 应用执行函数
- 将历史对话+函数执行结果再次发送给LLM
- LLM给出最终响应





>**User**: 
>query:今天上海天气如何?
>tool_description:
>	function_name:getweather
>	function_parameter: cityname(str)
>	function_desc:根据给定的程序名,返回对应城市的天气
>**Assistant**:
>answer:我将使用getweather进行上海天气查询
>tool_call:weather("上海")
>**User**:
>tool_call_result:上海天气阴....
>**Assistant**: 今天上海天气比较阴湿,能见度可能较低,建议外出时注意保暖并带上雨具。

**通过Function Call,大模型拥有了与现实世界交互的能力(数据获取、函数调用...)。**




#### MCP

MCP（Model Context Protocol，模型上下文协议） ，2024年11月底，由 Anthropic 推出的一种开放标准，旨在统一大型语言模型（LLM）与外部数据源和工具之间的通信协议。

不同的工具、不同的数据源,可能是不同的人使用不同的编程语言使用不同的方式进行调用,MCP的意图是将其统一起来。
无论什么样的数据、工具，只要按照MCP协议规范进行实现,即可进行统一调用。

![image-20250312155016470](MCP(Model%20Context%20Protocol)%20%E7%AE%80%E4%BB%8B/image-20250312155016470.png)

统一了函数(工具)的调用方式。


### 1. MCP Server Quick Start

大模型本身是无法获取到真实世界的数据的,这部分通过MCP构建一个获取天气的 MCP服务

#### 1.1 环境设置

```python
# uv安装( Python 包管理文件)
curl -LsSf https://astral.sh/uv/install.sh | sh


# Create a new directory for our project
uv init weather
cd weather

# Create virtual environment and activate it
uv venv
source .venv/bin/activate

# Install dependencies
uv add "mcp[cli]" httpx

# Create our server file
touch weather.py
```


#### 1.2 代码实现
核心代码逻辑编写，实现了 通过城市获取citycode ,通过citycode查看当前天气的功能

通过高德地图的api获取天气 https://console.amap.com/dev/key/app



```python
from typing import Any
import pandas as pd
import aiohttp
import asyncio
import json

def get_city_code(city_name:str)->str:
    """将用户输入的城市名转换为可供其他工具使用的城市编码"""
    df = pd.read_excel('citycode.xlsx', header=None)
    for index, row in df.iterrows():
        if city_name in row[0]:
            return row[1]
    return None

WEATHER_BASE_URL="https://restapi.amap.com/v3/weather/weatherInfo?key={}&city={}&extensions=base"
API_KEY="INPUT_API_KEY_HERE"

async def fetch_weather(city_code:str)->dict[str,Any]:
    url = WEATHER_BASE_URL.format(API_KEY, city_code)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                weather_data = await response.json()
                return weather_data
            else:
                return None

```


封装工具为MCP 服务
```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("weather")

@mcp.tool()
async def get_weather(city_code:str)->str:
    """ 通过给定的citycode获取到该城市对应的当前天气
        Args:
            citycode: 一个城市对应的citycode(如 440100 ,411426)
    """
    weather= await city2weather(city_code)
    if "lives" in weather and len(weather["lives"]) > 0:
        live = weather["lives"][0]
        formatted_weather = (
            f"省份: {live['province']}\n"
            f"城市: {live['city']}\n"
            f"天气: {live['weather']}\n"
            f"温度: {live['temperature']}°C\n"
            f"风向: {live['winddirection']}\n"
            f"风力: {live['windpower']}\n"
            f"湿度: {live['humidity']}%\n"
            f"报告时间: {live['reporttime']}"
        )
        return formatted_weather
    else:
        return "无法获取到对应的天气! 请检查citycode或城市名是否出错"

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
```
#### 1.3 测试运行& 导入 claude Desktop
运行下面代码，不报错代表服务没问题

```python
uv run .\apitest.py
```

编辑配置文件`claude_desktop_config.json` 让claude Desktop知道如何访问MCP服务

```json
{

    "mcpServers": {
        "weather": {
            "command": "uv",
            "args": [
                "--directory",
                "F:/A_Zhang/Code/MCPweather/weather",
                "run",
                "apitest.py"
            ]
        }
    }
}
```

重启Claude,它会自动运行这个 Python的 MCP服务
![image-20250312155047087](MCP(Model%20Context%20Protocol)%20%E7%AE%80%E4%BB%8B/image-20250312155047087.png)

问答测试
![image-20250312155058501](MCP(Model%20Context%20Protocol)%20%E7%AE%80%E4%BB%8B/image-20250312155058501.png)

#### 1.4 流程总结
将MCP绑定到claude Desktop(APP)后,用户输入需要调用 MCP服务的消息，整个过程如下:
- 客户端初始化与MCP的连接,获取可用的MCP服务(工具)
- 客户端(app)将 `用户问题` 和`可用的MCP服务` 发送给Claude
```python
        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages, # 用户的输入
            tools=available_tools # 可用的工具
        )
```

- Claude分析问题,并决定使用哪一个 MCP服务, 将要调用的服务与参数返回客户端(App)。
- App执行Claude选定的MCP 服务, 结果发送给Claude
  ![image-20250312155110164](MCP(Model%20Context%20Protocol)%20%E7%AE%80%E4%BB%8B/image-20250312155110164.png)
- Claude继续生成内容，最终呈现给用户。


### 2. MCP client Quick Start

环境配置
```python
# Create project directory
uv init mcp-client
cd mcp-client

# Create virtual environment
uv venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate

# Install required packages
uv add mcp anthropic python-dotenv


# Create our main file
touch client.py
```


在项目目录下创建 `.env`文件,将 APIkey粘贴到.env中
`ANTHROPIC_API_KEY=<your key here>`
#### 2.1 编写连接 init

client 初始化时,需要两个关键信息
- session -> 负责与MCP Server建立连接,用于查询、调用可用的工具
- Language Model -> LLM
- 
```python
class MCPClient:

    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.messages=[]
        self.anthropic = Anthropic(base_url="https://api.vveai.com")
        print(self.anthropic.api_key,)
        print(self.anthropic.base_url)
```


#### 2.2 连接Server实现
- 创建一个进程、运行MCP Server
- 将其封装成SESSION
```python
    async def connect_to_server(self,server_script_path:str):
        #  执行这个命令: uv --directory F:/A_Zhang/Code/MCPlearn/weather run apitest.py
        server_params = StdioServerParameters(
            command="uv",
            args=[
                "--directory",
                "F:/A_Zhang/Code/MCPlearn/weather",
                "run",
                server_script_path
            ],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        # 创建与 MCP Server的会话
        await self.session.initialize()


        #测试 获取MCP Server 中可用的 工具，打印出来
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
```


#### 2.3 处理用户输入实现
对用户输入的问题,调用大模型进行处理,大模型可以选择使用工具进行辅助

```python
    async def process_query(self, query: str) -> None:
        # 将用户输入处理成供API调用的格式
        self.messages.append(
            {
                "role": "user",
                "content": query
            }
        )
        # 从MCP Server 中获取可用的工具
        mcp_response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in mcp_response.tools]


        # 将用户问题、可用工具发送给LLM
        llm_response = self.anthropic.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=self.messages, # 用户的输入
            tools=available_tools # 可用的工具
        )
        
        # 如果响应中包含Function call,表示大模型需要调用工具,这个demo需要两次调用,所以使用While循环直到大模型不需要调用
        while llm_response.content[-1].type=="tool_use":
            # 更新记忆
            tool_content=llm_response.content[-1]
            self.messages.append({
                "role": "assistant",
                "content":  [tool_content]
            })

            #提取Function Call 中要调用的函数名、具体参数
            tool_name = tool_content.name
            tool_args = tool_content.input

            # 调用 MCP Server 对应的函数
            result = await self.session.call_tool(tool_name, tool_args)
            #  将函数调用结果更新到记忆中
            self.messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_content.id,
                        "content": result.content # 调用结果
                    }
                ]
            })

            # 将更新后的对话再次发送给大模型
            llm_response = self.anthropic.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=2048,
                messages=self.messages,
                tools=available_tools
            )
           
        # While循环结束,代表大模型不需要额外的MCP 调用,返回了最终结果
        print(llm_response.content[-1])
```


#### 2.4 交互接口 & 对象销毁
```python
async def chat_loop(self):
    """Run an interactive chat loop"""
    print("\nMCP Client Started!")
    print("Type your queries or 'quit' to exit.")

    while True:
        try:
            query = input("\nQuery: ").strip()

            if query.lower() == 'quit':
                break

            response = await self.process_query(query)
            print("\n" + response)

        except Exception as e:
            print(f"\nError: {str(e)}")

async def cleanup(self):
    """Clean up resources"""
    await self.exit_stack.aclose()
```


#### 2.5 main函数

```python
async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient() #初始化
    try:
        await client.connect_to_server(sys.argv[1])  #连接MCP Server
        await client.chat_loop() # 开始对话
    finally:
        await client.cleanup() # 清理

if __name__ == "__main__":
    import sys
    asyncio.run(main())
```











---

文章参考:

博客地址: qwrdxer.github.io

欢迎交流: qq1944270374