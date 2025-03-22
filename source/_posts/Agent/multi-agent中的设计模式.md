---
title: multi-agent中的设计模式
categories:
  - Agent
tags:
  - Agent
  - Multi-Agent
date: 2025-03-22 14:46:50
---



## Multi-Agent 设计模式

Agent 可以通过多种方式协同工作来解决问题,多Agent设计模式描述了多个Agent如何相互交互以解决实际的问题。


### Concurrent Agent 并发

本节介绍三种多Agent的并发模式
- Single Message & Multiple Processors ：即订阅同一Topic的多个Agent同时处理单个消息
- Multiple Message & Multiple Processors : 即如何根据topic将特定消息路由到专用Agent
- Direct Messaging : Agent与Agent之间如何发送消息

```python
import asyncio
from dataclasses import dataclass

from autogen_core import (
    AgentId,
    ClosureAgent,
    ClosureContext,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    default_subscription,
    message_handler,
    type_subscription,
)


@dataclass
class Task:
    task_id: str


@dataclass
class TaskResponse:
    task_id: str
    result: str

```


#### Single Message & Multiple Processors

- 示例中的每个Agent都使用 `default_subscription` 来订阅默认的topic
- 当将消息发送到默认的topic时，所有Agent都将独立处理该消息

示例中通过`publish_message`发送到默认topic，agent1和agent2都会处理这个任务
```python
@default_subscription
class Processor(RoutedAgent):

    @message_handler
    async def on_task(self,message:Task,ctx:MessageContext)->None:
        print(f"{self._description} starting task {message.task_id}")
        await asyncio.sleep(2)  # Simulate work
        print(f"{self._description} finished task {message.task_id}")

async def main():
    runtime = SingleThreadedAgentRuntime()
    await Processor.register(runtime,"agent1",lambda:Processor("agent1"))
    await Processor.register(runtime,"agent2",lambda:Processor("agent2"))
    runtime.start()
    await runtime.publish_message(Task(task_id="2"),DefaultTopicId())
    await runtime.stop_when_idle()
  

if __name__ == "__main__":
    asyncio.run(main())
```

#### Multiple messages & Multiple Processors

- 示例中创建了两个agent分别订阅了 `urgent` 与 `normal`两个topic
- 两个agent的处理结果会发送到 `task_results` 这个topic
- 可以通过 ClosureAgent来收集这两个Agent的处理结果

```python
TASK_RESULTS_TOPIC_TYPE = "task-results"
task_results_topic_id = TopicId(type=TASK_RESULTS_TOPIC_TYPE, source="default")


@type_subscription(topic_type="urgent")
class UrgentProcessor(RoutedAgent):
    @message_handler
    async def on_task(self, message: Task, ctx: MessageContext) -> None:
        print(f"Urgent processor starting task {message.task_id}")
        await asyncio.sleep(1)  # Simulate work
        print(f"Urgent processor finished task {message.task_id}")
	    #将处理结果发给指定的topic
        task_response = TaskResponse(task_id=message.task_id, result="Results by Urgent Processor")
        await self.publish_message(task_response, topic_id=task_results_topic_id)


@type_subscription(topic_type="normal")
class NormalProcessor(RoutedAgent):
    @message_handler
    async def on_task(self, message: Task, ctx: MessageContext) -> None:
        print(f"Normal processor starting task {message.task_id}")
        await asyncio.sleep(3)  # Simulate work
        print(f"Normal processor finished task {message.task_id}")

        task_response = TaskResponse(task_id=message.task_id, result="Results by Normal Processor")
        await self.publish_message(task_response, topic_id=task_results_topic_id)

```


- 注册Agent并向主题发送消息,Agent会去处理
```python
runtime = SingleThreadedAgentRuntime()

await UrgentProcessor.register(runtime, "urgent_processor", lambda: UrgentProcessor("Urgent Processor"))
await NormalProcessor.register(runtime, "normal_processor", lambda: NormalProcessor("Normal Processor"))

runtime.start()

await runtime.publish_message(Task(task_id="normal-1"), topic_id=TopicId(type="normal", source="default"))
await runtime.publish_message(Task(task_id="urgent-1"), topic_id=TopicId(type="urgent", source="default"))

await runtime.stop_when_idle()

```


- 然后使用 `ClosureAgent`来收集发布的信息
```python
queue = asyncio.Queue[TaskResponse]()


async def collect_result(_agent: ClosureContext, message: TaskResponse, ctx: MessageContext) -> None:
    await queue.put(message)


runtime.start()

CLOSURE_AGENT_TYPE = "collect_result_agent"
await ClosureAgent.register_closure(
    runtime,
    CLOSURE_AGENT_TYPE,
    collect_result,
    subscriptions=lambda: [TypeSubscription(topic_type=TASK_RESULTS_TOPIC_TYPE, agent_type=CLOSURE_AGENT_TYPE)],
)

await runtime.publish_message(Task(task_id="normal-1"), topic_id=TopicId(type="normal", source="default"))
await runtime.publish_message(Task(task_id="urgent-1"), topic_id=TopicId(type="urgent", source="default"))

await runtime.stop_when_idle()
while not queue.empty():
    print(await queue.get())
```


#### Direct Messages

- 有两种直接发送的方法,Agent之间发送或者通过runtime发送给指定Agent
- 消息使用AgentId进行寻址
- 发送者可能会期望收到来自目标Agent的Response
- **注册Agent是指注册了这个Agent类型 type**
- 当使用 `AgentId(type,key)` 传送消息时，运行时将获取实例或创建一个实例（如果不存在）。



### Sequential Workflow 工作流

顺序工作流是一种Multi Agent设计模式，其中Agent按确定性顺序响应。工作流程中的每个Agent通过处理消息、生成响应，然后将其传递给下一个Agent来执行特定任务。此模式可用于创建确定性工作流，其中每个Agent都为预先指定的子任务做出贡献。

- 每一个Agent监听一个topic
- 上一个Agent的输出会传给下一个topic
![image-20250322144834255](multi-agent%E4%B8%AD%E7%9A%84%E8%AE%BE%E8%AE%A1%E6%A8%A1%E5%BC%8F/image-20250322144834255.png)
实现思路: 
1. 首先画好整个工作流 ，每一个Agent的位置
2. 然后创建topic，用于Agent之间的通信
3. 将topic绑定给Agent
4. 发送消息给最初始的Agent所订阅的topic，整个工作流开始运作。

### Group Chat 群聊
群聊设计模式: 一组Agent共享一个公共的Messages，即他们都订阅和发布到同一个topic，每个参与Agent都专门用于特定任务，例如协作写作任务中的作家、插画家和编辑。您还可以包括一个代理来代表人类用户，以便在需要时帮助指导Agent。

- 在群聊中, 参与Agent会轮流发布消息，轮次的顺序由具体需求进行实现,可以使用循环算法或者让带有LLM的selector去选择
- 群聊可用于将复杂任务动态分解为较小的任务，这些任务可以由具有明确定义角色的专门代理处理。还可以将群聊嵌套到一个层次结构中，每个参与者都有一个递归群聊。

- ![image-20250322144823149](multi-agent%E4%B8%AD%E7%9A%84%E8%AE%BE%E8%AE%A1%E6%A8%A1%E5%BC%8F/image-20250322144823149.png)首先是topic设置, 每一个Agent都需要订阅两个Topic:一个用于接收公共消息,一个用于接收自己的任务消息
	- 任务消息由manager发布,公共消息是协作Agent的任务执行结果
- 我们需要定义一个Group Chat Manager Agent，用于 决定 由哪个Agent进行消息发布
- 协作Agent接收到自己的任务消息时会执行任务，接收到公共消息时，将其记录下来



### handsoffs交接

Handoff 是 OpenAI 在一个名为 [Swarm](https://github.com/openai/swarm) 的实验项目中引入的一种多代理设计模式。**关键思想是让代理使用特殊工具调用将任务委派给其他代理。** ，任务转交的本质还是工具调用。

![image-20250322144808738](multi-agent%E4%B8%AD%E7%9A%84%E8%AE%BE%E8%AE%A1%E6%A8%A1%E5%BC%8F/image-20250322144808738.png)

#### message

```python
class UserLogin(BaseModel):
    pass

# 用户的输入，由User Agent代替转发
class UserTask(BaseModel):
    context: List[LLMMessage]

# Agent的响应消息,
class AgentResponse(BaseModel):
    reply_to_topic_type: str
    context: List[LLMMessage]

```

#### Agent 
- 在Swarm中,Agent是不按照顺序进行轮流发言的，所以除了任务执行外，我们还需要定义一个特别的User Agent用于代替用户进行发言
- 功能性Agent中,定义了 sales  issue triage  human 这四个Agent用于处理用户问题

#### topic
每一个Agent都有一个自己的topic , Agent之间、Agent与用户的交互通过topic进行消息传递
```python
sales_agent_topic_type = "SalesAgent"
issues_and_repairs_agent_topic_type = "IssuesAndRepairsAgent"
triage_agent_topic_type = "TriageAgent"
human_agent_topic_type = "HumanAgent"
user_topic_type = "User"

```

#### 转发工具
转发的本质还是Function Call, 负责任务执行的Agent会自己判断是否需要转发,如果需要,则调用转发工具,这个工具会返回转发目标对应的Topic
```python
def transfer_to_sales_agent() -> str:
    return sales_agent_topic_type


def transfer_to_issues_and_repairs() -> str:
    return issues_and_repairs_agent_topic_type


def transfer_back_to_triage() -> str:
    return triage_agent_topic_type


def escalate_to_human() -> str:
    return human_agent_topic_type


transfer_to_sales_agent_tool = FunctionTool(
    transfer_to_sales_agent, description="Use for anything sales or buying related."
)
transfer_to_issues_and_repairs_tool = FunctionTool(
    transfer_to_issues_and_repairs, description="Use for issues, repairs, or refunds."
)
transfer_back_to_triage_tool = FunctionTool(
    transfer_back_to_triage,
    description="Call this if the user brings up a topic outside of your purview,\nincluding escalating to human.",
)
escalate_to_human_tool = FunctionTool(escalate_to_human, description="Only call this if explicitly asked to.")

```

#### UserAgent
在Swarm中,Agent是不按照顺序进行轮流发言的，所以除了任务执行外，我们还需要定义一个特别的User Agent用于代替用户进行发言
- handle_user_login用户一开始获取到用户输入的任务，将其发送给第一个代理 `triage_agent`
- handle_user_result:接收其他Agent的结果,获取用户输入,发给目标Agent进行下一步任务执行
```python
class UserAgent(RoutedAgent):
    def __init__(self, description: str, user_topic_type: str, agent_topic_type: str) -> None:
        super().__init__(description)
        self._user_topic_type = user_topic_type
        self._agent_topic_type = agent_topic_type

    @message_handler
    async def handle_user_login(self, message: UserLogin, ctx: MessageContext) -> None:
        print(f"{'-'*80}\nUser login, session ID: {self.id.key}.", flush=True)
        # Get the user's initial input after login.
        user_input = input("User: ")
        print(f"{'-'*80}\n{self.id.type}:\n{user_input}")
        await self.publish_message(
            UserTask(context=[UserMessage(content=user_input, source="User")]),
            topic_id=TopicId(self._agent_topic_type, source=self.id.key),
        )

    @message_handler
    async def handle_task_result(self, message: AgentResponse, ctx: MessageContext) -> None:
        # Get the user's input after receiving a response from an agent.
        user_input = input("User (type 'exit' to close the session): ")
        print(f"{'-'*80}\n{self.id.type}:\n{user_input}", flush=True)
        if user_input.strip().lower() == "exit":
            print(f"{'-'*80}\nUser session ended, session ID: {self.id.key}.")
            return
        message.context.append(UserMessage(content=user_input, source="User"))
        await self.publish_message(
            UserTask(context=message.context), topic_id=TopicId(message.reply_to_topic_type, source=self.id.key)
        )

```



#### 任务执行Agent
其需要执行用户发送的Task,具体步骤如下
1. 调用client 对用户的输入进行任务执行
2. 如果有工具调用，则进入循环
	1. 运行所有的工具调用，将任务工具调用结果、代理转发工具结果分别存储
	2. 如果有转发工具的调用结果,则将其转发给目标
	3. 如果有任务工具的调用结果，则调用client继续执行任务，否则代表只有转发的tool call，直接结束本次任务
3. 如果到了这里，则代表Agent已经执行全部的任务，将结果返回给用户即可

总体来看,任务执行的Agent有两个目标: 任务执行给用户响应、任务转发,
```python

class AIAgent(RoutedAgent):
    def __init__(
        self,
        description: str,
        system_message: SystemMessage,
        model_client: ChatCompletionClient,
        tools: List[Tool],
        delegate_tools: List[Tool],
        agent_topic_type: str,
        user_topic_type: str,
    ) -> None:
        super().__init__(description)
        self._system_message = system_message
        self._model_client = model_client
        self._tools = dict([(tool.name, tool) for tool in tools])
        self._tool_schema = [tool.schema for tool in tools]
        self._delegate_tools = dict([(tool.name, tool) for tool in delegate_tools])
        self._delegate_tool_schema = [tool.schema for tool in delegate_tools]
        self._agent_topic_type = agent_topic_type
        self._user_topic_type = user_topic_type

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> None:
        #让大模型进行补全，同时提供代理工具,用于大模型进行权限移交.
        llm_result = await self._model_client.create(
            messages=[self._system_message] + message.context,
            tools=self._tool_schema + self._delegate_tool_schema,
            cancellation_token=ctx.cancellation_token,
        )
        print(f"{'-'*80}\n{self.id.type}:\n{llm_result.content}", flush=True)
        # Process the LLM result. 处理工具调用
        while isinstance(llm_result.content, list) and all(isinstance(m, FunctionCall) for m in llm_result.content):
            tool_call_results: List[FunctionExecutionResult] = []
            delegate_targets: List[Tuple[str, UserTask]] = []
            # Process each function call.
            for call in llm_result.content:
                arguments = json.loads(call.arguments)
                if call.name in self._tools: #正常的工具调用
                    # Execute the tool directly.
                    result = await self._tools[call.name].run_json(arguments, ctx.cancellation_token)
                    result_as_str = self._tools[call.name].return_value_as_string(result)
                    tool_call_results.append(
                        FunctionExecutionResult(call_id=call.id, content=result_as_str, is_error=False, name=call.name)
                    )
                elif call.name in self._delegate_tools: #代理工具调用
                    # Execute the tool to get the delegate agent's topic type.
                    result = await self._delegate_tools[call.name].run_json(arguments, ctx.cancellation_token)
                    topic_type = self._delegate_tools[call.name].return_value_as_string(result)
                    # Create the context for the delegate agent, including the function call and the result.
                    delegate_messages = list(message.context) + [
                        AssistantMessage(content=[call], source=self.id.type),
                        FunctionExecutionResultMessage(
                            content=[
                                FunctionExecutionResult(
                                    call_id=call.id,
                                    content=f"Transferred to {topic_type}. Adopt persona immediately.",
                                    is_error=False,
                                    name=call.name,
                                )
                            ]
                        ),
                    ]
                    delegate_targets.append((topic_type, UserTask(context=delegate_messages)))
                else:
                    raise ValueError(f"Unknown tool: {call.name}")
            if len(delegate_targets) > 0:
                # Delegate the task to other agents by publishing messages to the corresponding topics.
                for topic_type, task in delegate_targets:
                    print(f"{'-'*80}\n{self.id.type}:\nDelegating to {topic_type}", flush=True)
                    await self.publish_message(task, topic_id=TopicId(topic_type, source=self.id.key))
            if len(tool_call_results) > 0:
                print(f"{'-'*80}\n{self.id.type}:\n{tool_call_results}", flush=True)
                # Make another LLM call with the results.
                message.context.extend(
                    [
                        AssistantMessage(content=llm_result.content, source=self.id.type),
                        FunctionExecutionResultMessage(content=tool_call_results),
                    ]
                )
                llm_result = await self._model_client.create(
                    messages=[self._system_message] + message.context,
                    tools=self._tool_schema + self._delegate_tool_schema,
                    cancellation_token=ctx.cancellation_token,
                )
                print(f"{'-'*80}\n{self.id.type}:\n{llm_result.content}", flush=True)
            else:
                # The task has been delegated, so we are done.
                return
        # The task has been completed, publish the final result.
        assert isinstance(llm_result.content, str)
        message.context.append(AssistantMessage(content=llm_result.content, source=self.id.type))
        await self.publish_message(
            AgentResponse(context=message.context, reply_to_topic_type=self._agent_topic_type),
            topic_id=TopicId(self._user_topic_type, source=self.id.key),
        )

```

- 工具分为两种，功能工具和代理工具 ，功能工具是实现Agent 功能的工具，代理工具则负责转发
- Agent在调用llm client时，会将这两种工具发送给大模型，让其决定工具的使用
- 如果LLM client决定转发，则构造相应的message发送到相应的topic



#### Mixture of Agent

https://github.com/togethercomputer/moa

![image-20250322144753518](multi-agent%E4%B8%AD%E7%9A%84%E8%AE%BE%E8%AE%A1%E6%A8%A1%E5%BC%8F/image-20250322144753518.png)

混合代理是一种 multi-Agent设计模式,它按照前馈神经网络架构进行建模
- 该模式有两种Agent组成: worker Agent与 orchestrator Agent
- workerAgent被按照固定数量分为多个层
- orchestrator agent 接收用户输入,并将其发送给第一层中的worker Agent
- 每一个worker Agent都执行任务 
- orchestrator agent聚合任务结果，并将包含先前结果的更新任务分派给第二层中的工作代理。
- 在最后一层中，orchestrator agent 聚合上一层的结果，并将单个最终结果返回给用户。

```python
class OrchestratorAgent(RoutedAgent):
    def __init__(
        self,
        model_client: ChatCompletionClient,
        worker_agent_types: List[str],
        num_layers: int,
    ) -> None:
        super().__init__(description="Aggregator Agent")
        self._model_client = model_client
        self._worker_agent_types = worker_agent_types
        self._num_layers = num_layers

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> FinalResult:
        print(f"{'-'*80}\nOrchestrator-{self.id}:\nReceived task: {message.task}")
        # Create task for the first layer.
        worker_task = WorkerTask(task=message.task, previous_results=[])
        # Iterate over layers.
        for i in range(self._num_layers - 1):
            # Assign workers for this layer.
            worker_ids = [
                AgentId(worker_type, f"{self.id.key}/layer_{i}/worker_{j}")
                for j, worker_type in enumerate(self._worker_agent_types)
            ]
            # Dispatch tasks to workers.
            print(f"{'-'*80}\nOrchestrator-{self.id}:\nDispatch to workers at layer {i}")
            results = await asyncio.gather(*[self.send_message(worker_task, worker_id) for worker_id in worker_ids])
            print(f"{'-'*80}\nOrchestrator-{self.id}:\nReceived results from workers at layer {i}")
            # Prepare task for the next layer.
            worker_task = WorkerTask(task=message.task, previous_results=[r.result for r in results])
        # Perform final aggregation.
        print(f"{'-'*80}\nOrchestrator-{self.id}:\nPerforming final aggregation")
        system_prompt = "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n\nResponses from models:"
        system_prompt += "\n" + "\n\n".join([f"{i+1}. {r}" for i, r in enumerate(worker_task.previous_results)])
        model_result = await self._model_client.create(
            [SystemMessage(content=system_prompt), UserMessage(content=message.task, source="user")]
        )
        assert isinstance(model_result.content, str)
        return FinalResult(result=model_result.content)

```


#### Multi-Agent Debate

多智能体辩论是一种多智能体设计模式，用于模拟多轮次交互，其中在每个轮次中，智能体相互交换响应，并根据其他智能体的响应优化其响应。
https://arxiv.org/abs/2406.11776


该模式的工作原理如下：

1. 用户向聚合器代理发送数学问题。
   
2. 聚合器代理将问题分配给求解器代理。
   
3. 每个求解器代理都会处理问题，并向其邻居发布响应。
   
4. 每个求解器代理都使用来自其邻居的响应来优化其响应，并发布新的响应。
   
5. 重复步骤 4 固定轮数。在最后一轮中，每个求解器代理都会发布最终响应。
   
6. 聚合器代理使用多数投票来聚合来自所有求解器代理的最终响应以获得最终答案，然后发布答案。

求解器处理两种消息，一种是邻居的消息，当期接收到足够多的邻居消息时，会进行问题求解，随后转发给邻居


#### Reflection

![image-20250322144732954](multi-agent%E4%B8%AD%E7%9A%84%E8%AE%BE%E8%AE%A1%E6%A8%A1%E5%BC%8F/image-20250322144732954.png)
反射可以使用两个Agent进行实现，第一个Agent负责生成任务执行结果,第二个Agent则对任务执行结果进行响应，两个Agent继续交互，直到达到终止条件(最大迭代次数 或者 第二个Agent批准)

- 有四种消息类型,任务输入、结果输出 、 任务中间结果、任务中间结果评价
- Agent需要有记忆力



---

文章参考:

博客地址: qwrdxer.github.io

欢迎交流: qq1944270374