---
title: mongodb学习
toc: true
categories:
  - 数据库
date: 2022-04-14 09:31:33
tags:
---



## 安装MongoDB

### docker 安装

启动docker  --auth表示需要授权才能连接

```docker
docker run -itd --name mongo -p 27017:27017 mongo --auth
```

添加密码

```shell
$ docker exec -it mongo mongo admin
# 创建一个名为 admin，密码为 123456 的用户。
>  db.createUser({ user:'admin',pwd:'qwrdxer',roles:[ { role:'userAdminAnyDatabase', db: 'admin'},"readWriteAnyDatabase"]});
# 尝试使用上面创建的用户信息进行连接。
> db.auth('admin', 'qwrdxer')
```

## MongDB基本概念

| SQL术语/概念 | MongoDB术语/概念 | 解释/说明                           |
| :----------- | :--------------- | :---------------------------------- |
| database     | database         | 数据库                              |
| table        | collection       | 数据库表/集合                       |
| row          | document         | 数据记录行/文档                     |
| column       | field            | 数据字段/域                         |
| index        | index            | 索引                                |
| table joins  |                  | 表连接,MongoDB不支持                |
| primary key  | primary key      | 主键,MongoDB自动将_id字段设置为主键 |

### 数据库

一个mongodb中可以建立多个数据库。

MongoDB的默认数据库为"db"，该数据库存储在data目录中。

MongoDB的单个实例可以容纳多个独立的数据库，每一个都有自己的集合和权限，不同的数据库也放置在不同的文件中。



有一些数据库名是保留的，可以直接访问这些有特殊作用的数据库。

- **admin**： 从权限的角度来看，这是"root"数据库。要是将一个用户添加到这个数据库，这个用户自动继承所有数据库的权限。一些特定的服务器端命令也只能从这个数据库运行，比如列出所有的数据库或者关闭服务器。
- **local:** 这个数据永远不会被复制，可以用来存储限于本地单台服务器的任意集合
- **config**: 当Mongo用于分片设置时，config数据库在内部使用，用于保存分片的相关信息。



*<!-- more -->* 

### 文档

文档是一组键值(key-value)对(即 BSON)。MongoDB 的文档不需要设置相同的字段，并且相同的字段不需要相同的数据类型，这与关系型数据库有很大的区别，也是 MongoDB 非常突出的特点。

1. 文档中的键/值对是有序的。
2. 文档中的值不仅可以是在双引号里面的字符串，还可以是其他几种数据类型（甚至可以是整个嵌入的文档)。
3. MongoDB区分类型和大小写。
4. MongoDB的文档不能有重复的键。
5. 文档的键是字符串。除了少数例外情况，键可以使用任意UTF-8字符。

### 集合

集合就是 MongoDB 文档组，类似于 RDBMS （关系数据库管理系统：Relational Database Management System)中的表格。

集合存在于数据库中，集合没有固定的结构，这意味着你在对集合可以插入不同格式和类型的数据，但通常情况下我们插入集合的数据都会有一定的关联性。



比如，我们可以将以下不同数据结构的文档插入到集合中：

```
{"site":"www.baidu.com"}
{"site":"www.google.com","name":"Google"}
{"site":"www.runoob.com","name":"菜鸟教程","num":5}
```

当第一个文档插入时，集合就会被创建。



### MongoDB 数据类型

下表为MongoDB中常用的几种数据类型。

| 数据类型           | 描述                                                         |
| :----------------- | :----------------------------------------------------------- |
| String             | 字符串。存储数据常用的数据类型。在 MongoDB 中，UTF-8 编码的字符串才是合法的。 |
| Integer            | 整型数值。用于存储数值。根据你所采用的服务器，可分为 32 位或 64 位。 |
| Boolean            | 布尔值。用于存储布尔值（真/假）。                            |
| Double             | 双精度浮点值。用于存储浮点值。                               |
| Min/Max keys       | 将一个值与 BSON（二进制的 JSON）元素的最低值和最高值相对比。 |
| Array              | 用于将数组或列表或多个值存储为一个键。                       |
| Timestamp          | 时间戳。记录文档修改或添加的具体时间。                       |
| Object             | 用于内嵌文档。                                               |
| Null               | 用于创建空值。                                               |
| Symbol             | 符号。该数据类型基本上等同于字符串类型，但不同的是，它一般用于采用特殊符号类型的语言。 |
| Date               | 日期时间。用 UNIX 时间格式来存储当前日期或时间。你可以指定自己的日期时间：创建 Date 对象，传入年月日信息。 |
| Object ID          | 对象 ID。用于创建文档的 ID。                                 |
| Binary Data        | 二进制数据。用于存储二进制数据。                             |
| Code               | 代码类型。用于在文档中存储 JavaScript 代码。                 |
| Regular expression | 正则表达式类型。用于存储正则表达式。                         |

## 操控MongoDB

### 基本操作

> 数据库操作

```mysql
show dbs 查看存在的数据库
db 查看当前数据库
use  xxx 切换到指定数据库 ,同时若数据库不存在也会创建一个数据库

db.dropDatabase() 删除当前数据库


```

> 集合操作

```mysql
db.createCollection(name, options)
参数说明：
- name: 要创建的集合名称
- options: 可选参数, 指定有关内存大小及索引的选项


db.collection.drop("target")删除指定的集合
show collections 、show tables 已有集合
```



---

>  插入文档

db.collection.insertOne() 用于向集合插入一个新文档，语法格式如下：

```
db.collection.insertOne(
   <document>,
   {
      writeConcern: <document>
   }
)
```

db.collection.insertMany() 用于向集合插入一个多个文档，语法格式如下：

```
db.collection.insertMany(
   [ <document 1> , <document 2>, ... ],
   {
      writeConcern: <document>,
      ordered: <boolean>
   }
)

document：要写入的文档。
writeConcern：写入策略，默认为 1，即要求确认写操作，0 是不要求。
ordered：指定是否按顺序写入，默认 true，按顺序写入。
```



---

>  查看文档

```
db.collection.find(query, projection)
```

- **query** ：可选，使用查询操作符指定查询条件
- **projection** ：可选，使用投影操作符指定返回的键。查询时返回文档中所有键值， 只需省略该参数即可（默认省略）

通过 limit 函数限制查询命令返回的文档数量，命令格式如下：

**db.目标集合.find().limit( 数量 )**

即查询返回指定“数量”的文档数据。

---

>  更新文档

update() 方法用于更新已存在的文档。语法格式如下：

```
db.collection.update(
   <query>,
   <update>,
   {
     upsert: <boolean>,
     multi: <boolean>,
     writeConcern: <document>
   }
```

**参数说明：**

- **query** : update的查询条件，类似sql update查询内where后面的。
- **update** : update的对象和一些更新的操作符（如$,$inc...）等，也可以理解为sql update查询内set后面的
- **upsert** : 可选，这个参数的意思是，如果不存在update的记录，是否插入objNew,true为插入，默认是false，不插入。
- **multi** : 可选，mongodb 默认是false,只更新找到的第一条记录，如果这个参数为true,就把按条件查出来多条记录全部更新。
- **writeConcern** :可选，抛出异常的级别。

操作符: https://wenku.baidu.com/view/ac7a7f80f624ccbff121dd36a32d7375a417c6d8.html



> 删除文档

remove() 方法的基本语法格式如下所示：

```
db.collection.remove(
   <query>,
   {
     justOne: <boolean>,
     writeConcern: <document>
   }
)
```

**参数说明：**

- **query** :（可选）删除的文档的条件。
- **justOne** : （可选）如果设为 true 或 1，则只删除一个文档，如果不设置该参数，或使用默认值 false，则删除所有匹配条件的文档。
- **writeConcern** :（可选）抛出异常的级别。





> 通过URI连接MongoDB

```mysql
mongodb://[username:password@]host1[:port1][,host2[:port2],...[,hostN[:portN]]][/[database][?options]]
```

- **mongodb://** 这是固定的格式，必须要指定。
- **username:password@** 可选项，如果设置，在连接数据库服务器之后，驱动都会尝试登录这个数据库
- **host1** 必须的指定至少一个host, host1 是这个URI唯一要填写的。它指定了要连接服务器的地址。如果要连接复制集，请指定多个主机地址。
- **portX** 可选的指定端口，如果不填，默认为27017
- **/database** 如果指定username:password@，连接并验证登录指定数据库。若不指定，默认打开 test 数据库。
- **?options** 是连接选项。如果不使用/database，则前面需要加上/。所有连接选项都是键值对name=value，键值对之间通过&或;（分号）隔开





### 添加文档

```mysql
db.tasklist.insert({
	taskUID: "task2022-4-14-1",
	task_domain: {
		domain_total: 3,
		domain_list: [{
			domain: "www.baidu.com",
			status: 1
		}, {
			domain: "www.4399.com",
			status: 2
		}, {
			domain: "www.kkk.com"
		}]
	},
	task_ip: {
		ip_total: 1,
		ip_list: [{
			ip: "123.123.123.123",
			port_info: [{
				port: 80,
				finger: "web"
			}, {
				port: 8080,
				finger: "web"
			}]
		}, {
			ip: "123.123.123.123",
			port_info: [{
				port: 8081,
				finger: "web"
			}]
		}]
	}
})
```



### 文档中追加数据

追加一条IP

```mysql
db.tasklist.update({taskUID: "task2022-4-14-1"},{$push:{task_ip.ip_list}})
```



### 修改文档中的值



### 对文档进行条件查询

查询未扫描的域名

```mysql
db.tasklist.aggregate([
    {"$match":{
    "taskUID" : "task2022-4-14-1"
    }},
    {"$unwind":"$task_domain"},
    {"$match":{
    	"task_domain.domain_list.status":1
    }},
    {"$group":{
    	"_id":"$_id",
    	"task_domain":{"$push":"$task_domain"}
    }}
])
```

```mysql
db.tasklist.find({"task_domain.domain_list.status":1},{'task_domain.domain_list.status':1})
```



## Go语言操控MongoDB



*<!-- more -->* 







> 参考文章
>
> MongoDB中如何为集合的数组类型字段添加一个值
>
> https://jingyan.baidu.com/article/4b07be3c6b183809b280f34f.html