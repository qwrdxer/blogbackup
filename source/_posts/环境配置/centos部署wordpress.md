---
title: centos部署wordpress
date: 2021-04-15 01:34:16
tags:
- 环境配置
- 报错解决
- linux
- WordPress
categories:
- 环境配置
---

### 1.安装apache

```
sudo yum install httpd

# 启动httpd
sudo service httpd start
```

浏览器输入ip访问测试页面

*<!-- more -->* 

### 2.安装mysql

```bash
wget -i -c http://dev.mysql.com/get/mysql57-community-release-el7-10.noarch.rpm
yum -y install mysql57-community-release-el7-10.noarch.rpm
yum -y install mysql-community-server

#启动mysql
systemctl start  mysqld.service

#获得密码
grep "password" /var/log/mysqld.log

#使用查找到的密码登录
mysql -u root -p

# 登录成功后输入如下命令修改密码
> ALTER USER 'root'@'localhost' IDENTIFIED BY 'new password';


# 添加远程登录(可选则跳过)
grant all privileges on *.* to 'root'@'%' identified by 'password' with grant option;
flush privileges;
```

### 3.安装php(>5.6.20 )

```bash
rpm -Uvh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
rpm -Uvh https://mirror.webtatic.com/yum/el7/webtatic-release.rpm
yum -y install php71w-common php71w-fpm php71w-opcache php71w-gd php71w-mysqlnd php71w-mbstring php71w-pecl-redis php71w-pecl-memcached php71w-devel mod_php71w.x86_64
systemctl  start php-fpm
systemctl enable php-fpm
```

无法解析php文件参考这篇博客（后来知道是少安装了mod_php71w.x86_64)

https://blog.csdn.net/qq_33858250/article/details/81270278

https://zhuanlan.zhihu.com/p/126717388

### 4. 设置mysql、httpd开机自启动

```bash
sudo chkconfig httpd on
sudo chkconfig mysqld on
```

### 5. 下载安装wordpress

https://cn.wordpress.org/

或者 命令 wget https://cn.wordpress.org/latest-zh_CN.tar.gz

太慢了建议翻墙

```bash
 #解压到指定目录
 unzip wordpress-5.7-zh_CN.zip -d /var/www/html/


 # mysql中创建数据库
mysql -u root -p

create database wordpress;
# 退出mysql
exit

# 编辑配置文件
cd /var/www/html/wordpress
cp wp-config-sample.php wp-config.php

vim wp-config.php
#修改这几个字段
/** MySQL数据库名：wordpress */
define(‘DB_NAME', ‘wordpress'); 
/** MySQL数据库用户名 :root*/
define(‘DB_USER', ‘root'); 
/** MySQL数据库密码 :password*/
define(‘DB_PASSWORD', ‘your password');
/** MySQL主机（不用修改） */
define(‘DB_HOST', ‘localhost');
```

### 6.测试安装

浏览器登录即可

### 7.参考博客

https://blog.csdn.net/qq_36582604/article/details/80526287

https://www.cnblogs.com/DarrenChan/p/6622233.html