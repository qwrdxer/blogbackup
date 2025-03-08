---
title: some script
tags:
  - 脚本
  - python
  - bash
categories:
  - 脚本
date: 2021-04-15 11:23:33
---

## 记录一些有用的小工具

### 使用Python在当前文件下开启HTTP服务

Python <= 2.3

python -c "import SimpleHTTPServer as s; s.test();" 8000

Python >= 2.4

python -m SimpleHTTPServer 8000

Python 3.x

python3 -m http.server 8000

 python -m SimpleHTTPServer

使用python开启ftp

 python3 -m pyftpdlib -p 21

*<!-- more -->* 

### 分屏(在各大ssh连接中十分有用)

screen -S name 创建一个id为name的shellTerminal

screen -ls 列出所有Terminal

screen -r name 进入某个Terminal



### Snap安装

apt install snapd

systemctl enable --now snapd apparmor

snap install qv2ray



### 从一堆http链接中提取子域名

cat file | grep -v "http" >> outputurl

cat file | grep "http" >> output2url

cat output2url | awk -F "/" '{print $3}' > ouputurl



### 一键安装docker

curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun

也可以使用国内 daocloud 一键安装命令：

curl -sSL https://get.daocloud.io/docker | sh



### 正则表达式记录

匹配IP地址

((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)

grep  '\([0-9]\{1,3\}\.\)\{3\}[0-9]\{1,3\}' res.txt



### 提取 ip列表中的C段

cat ip.txt |sort|uniq|grep -E -o "([0-9]{1,3}[\.]){3}"|uniq -c| awk '{if ($1>=3) print $2"0/24"}' >ip2.txt

### 提取文件中域名

grep -ohr -E "https?://[a-zA-Z0-9\.\/_&=@$%?~#-]*" ./folder

### 给文件所有行添加字符串

cat ip_all_all.txt | awk '{print"http://"$0 }' >> ip_http.txt    



### 压缩文件

tar -czvf xxx.tar.gz file/





### 文件去重(取自《linux shell 脚本攻略》)

```shell
# !/bin/bash
# 文件名: remove_duplicates.sh
# 用途: 查找并删除重复文件，每一个文件只保留一份
ls -lS --time-style=long-iso | awk 'BEGIN {
getline; getline;
name1=$8; size=$5
}
{
name2=$8;
if (size==$5)
{
"md5sum "name1 | getline; csum1=$1;
"md5sum "name2 | getline; csum2=$1;
if ( csum1==csum2 )
{
print name1; print name2
}
};
size=$5; name1=name2;
}' | sort -u > duplicate_files
cat duplicate_files | xargs -I {} md5sum {} | \
sort | uniq -w 32 | awk '{ print $2 }' | \
sort -u > unique_files
echo Removing..
comm duplicate_files unique_files -3 | tee /dev/stderr | \
xargs rm
echo Removed duplicates files successfully.
```

### [shell 使用sed去除换行以及去除空格](https://www.cnblogs.com/zl1991/p/15181070.html)

去除换行：

sed ":a;N;s/\n//g;ta" result

去除所有空格

sed s/[[:space:]]//g result

windows换行符转linux换行符

单个的文件装换

sed -i 's/\r//'  filename