---
title: docker相关
date: 2021-04-20 13:49:50
tags:
- 工具使用
- docker
- 脚本
categories:
- 开发工具
---

## 安装docker

**卸载旧的docker：**

```
sudo yum remove docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-engine
```

**设置仓库,配置加速：**

```
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
```

**更新yum索引：**

```
yum makecache fast
```

**安装docker引擎**：

```
1.安装最新版本的docker引擎，容器
$ sudo yum install docker-ce docker-ce-cli contanerd.io
 ce：社区版 ee:企业版
```

**启动docker：**

```
sudo systemctl start docker
```

**基本测试：**

```
docker version
```

*<!-- more -->* 

## Docker基本命令

### 帮助命令

```
docker version # 显示docker的版本信息
docker info    # 显示docker的系统信息,包括镜像和容器的数量
docker 命令 --help# 帮助命令
```

### 镜像命令

**docker images :**

```
root@iZiiwad3d3m7zzZ ~# docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
hello-world         latest              bf756fb1ae65        4 months ago        13.3kB

#解释
REPOSITORY 镜像的仓库源
TAG        镜像的标签
IMAGE ID   镜像的ID
CREATED    镜像创建的时间 
SIZE       镜像的大小

# 可选项

Options:
  -a, --all             Show all images (default hides intermediate images)
      --digests         Show digests

  -q, --quiet           Only show numeric IDs

# 示例：
docker images -aq 显示本地所有的镜像的id(后面可以用于批量管理的操作)
```

**docker search 搜索镜像**

```
docker serach mysql
NAME                              DESCRIPTION                                     STARS               OFFICIAL            AUTOMATED
mysql                             MySQL is a widely used, open-source relation…   9500                [OK]
mariadb                           MariaDB is a community-developed fork of MyS…   3444                [OK]

#根据关键词在仓库中搜索镜像并安stars数排序返回

#可选项 过滤
--filter=STARS=3000
```

**docker pull 下载镜像**

```
# docker pull 镜像名  默认下载最新镜像
# 下载指定版本的镜像: docker pull 镜像名[：tag] 


Pull an image or a repository from a registry

Options:
  -a, --all-tags                Download all tagged images in the repository
      --disable-content-trust   Skip image verification (default true)
  -q, --quiet                   Suppress verbose output


root@iZiiwad3d3m7zzZ ~# docker pull mysql
Using default tag: latest # 不写tag 默认最新
latest: Pulling from library/mysql
5b54d594fba7: Pull complete # 分层下载 docker image的核心 联合文件系统
07e7d6a8a868: Pull complete
abd946892310: Pull complete
dd8f4d07efa5: Pull complete
076d396a6205: Pull complete
cf6b2b93048f: Pull complete
530904b4a8b7: Pull complete
fb1e55059a95: Pull complete
4bd29a0dcde8: Pull complete
b94a001c6ec7: Pull complete
cb77cbeb422b: Pull complete
2a35cdbd42cc: Pull complete
Digest: sha256:dc255ca50a42b3589197000b1f9bab2b4e010158d1a9f56c3db6ee145506f625 # 签名
Status: Downloaded newer image for mysql:latest
docker.io/library/mysql:latest # 真实地址

docker pull mysql
等价于
docker pull  docker.io/library/mysql:latest

# 指定 版本下载:
docker pull mysql:5.7
```

**docker rmi 删除镜像**

```
#删除指定id镜像
docker rmi -f 镜像id[ 镜像id2 ...]
#删除全部镜像
docker rmi -f $(docker images -aq) 
```

### 容器命令

**说明:本地有了镜像后，才可以创建容器,这里使用一个centos镜像来测试学习**

```
docker pull centos
```

**新建容器并启动**

```
docker run [可选参数] images

# 参数说明
--name="centos1" 指定容器名字用来区分容器
-d 指定后台方式于宁
-it 使用交互方式运行 (进入容器)
-p 指定容器的端口
    使用方式:
    -p ip:主机端口:容器端口
    -p 主机端口:容器端口  (进行映射,使得外部可以通过指定的端口访问容器)
    -p 容器端口
    容器端口
-P 随机指定端口
 # 启动并进入容器
[root@iZiiwad3d3m7zzZ ~]# docker run -it centos /bin/bash
[root@b7c147e9497f /]# ls
bin  etc   lib    lost+found  mnt  proc  run   srv  tmp  var
dev  home  lib64  media       opt  root  sbin  sys  usr
# 通过ls可以看到这就是一个小型的centos环境
```

**退出容器**

```
从容器中退出
exit# 容器停止退出

Ctrl+P+Q #容器不停止退出
```

**列出所有运行中的容器**

```
docker ps
    默认为显示运行中的容器
    -a 查看创建的容器(包括没有在运行的)
    -n=? 显示最近创建的容器 n表示条数
    -q 仅显示创建的容器的id

docker ps -aq 查看所有创建的容器的id
```

**删除容器**

```
# 删除指定id的容器
docker rm 容器id # 不能删除正在运行中的容器 强制删除需带上参数 -f
# 递归删除所有的容器
docker rm -f $(docker ps -aq)
docker ps -aq | xargs docker rm 
```

**启动和停止容器的操作**

```
docker start id        #启动容器
docker restart id    #重启容器
docker stop         #停止运行中的容器
docker kill            #强制停止当前容器
```

### 常用其他命令

**后台启动容器**

```
#通过 -d 后台启动
[root@iZiiwad3d3m7zzZ ~]# docker run -d centos

# 发现centos停止了 

# 常见的坑: docker容器使用后台运行，必须要有一个前台进程,docker发现没有前台应用就会自动停止
# 容器启动后发现自己没有提供服务,就会立刻停止。
```

**查看日志**

```
docker logs
# 参数
Options:
      --details        Show extra details provided to logs
  -f, --follow         Follow log output
      --since string   Show logs since timestamp (e.g.
                       2013-01-02T13:23:37) or relative (e.g. 42m for 42
                       minutes)
      --tail string    Number of lines to show from the end of the logs
                       (default "all")
  -t, --timestamps     Show timestamps
      --until string   Show logs before a timestamp (e.g.
                       2013-01-02T13:23:37) or relative (e.g. 42m for 42
                       minutes)


 docker logs -f -t --tail 10 2072f1e07af8 # 查看当前日志
 # 后台运行一个bash命令，方便日志查看
 docker run -d centos /bin/sh -c "while true; do echo hahahahaha;sleep 1;done"
```

**查看容器中进程的信息**

```
docker top id

[root@iZiiwad3d3m7zzZ ~]# docker top 3c3546037cf0

UID                 PID                 PPID                C                   STIME               TTY                 TIME                CMD
root                15917               15869               0                   13:00               ?                   00:00:00            /bin/sh -c while true; do echo hahahahaha;sleep 1;done
root                16163               15917               0                   13:02               ?                   00:00:00            /usr/bin/coreutils --coreutils-prog-shebang=sleep /usr/bin/sleep 1
```

**查看镜像中的元数据**

```
docker inspect


Options:
  -f, --format string   Format the output using the given Go template
  -s, --size            Display total file sizes if the type is container
      --type string     Return JSON for specified type


[root@iZiiwad3d3m7zzZ ~]# docker inspect 3c3546037cf0
```

**进入当前正在运行的容器**

```
# 通常容器都是使用后台方式运行的  有时需要进入容器,修改配置

# 命令
docker exec it 容器 id bashshell
[root@iZiiwad3d3m7zzZ ~]# docker exec -it 749fdb83ed84 /bin/bash
[root@749fdb83ed84 /]#

#命令2
docker attach 容器 id


#docker exec 进入容器后打开一个新的终端
#docker attach 进入容器正在执行的终端，不会启动新的终端
```

**从容器中拷贝文件**

```
docker cp 容器id:容器内路径 主机路径

[root@iZiiwad3d3m7zzZ ~]# docker cp 749fdb83ed84:/home/1.txt ./
[root@iZiiwad3d3m7zzZ ~]# ls
1.txt  demo
*
```

**docker 查看cpu占用**

```
docker stats
```

## 使用数据卷

> 方式1：直接使用命令挂载

```
docker run -it -v 宿主机的目录:容器内的目录 镜像名 /bin/bash
#举例
[root@iZiiwad3d3m7zzZ ~]# docker run -it -v /home/test:/home centos /bin/bash

#使用ctrl pq 退出容器  ps查看容器正在运行
[root@51b1326faaa7 /]# [root@iZiiwad3d3m7zzZ ~]# docker ps
CONTAINER ID        IMAGE                 COMMAND                  CREATED             STATUS              PORTS                                            NAMES
51b1326faaa7        centos                "/bin/bash"              10 seconds ago      Up 

# 进入主机的挂载目录,创建一个 test1.txt 并输入一些内容
[root@iZiiwad3d3m7zzZ ~]# cd /home
[root@iZiiwad3d3m7zzZ home]#
[root@iZiiwad3d3m7zzZ home]# ls
1.txt  test
[root@iZiiwad3d3m7zzZ home]# cd test/
[root@iZiiwad3d3m7zzZ test]# ls
[root@iZiiwad3d3m7zzZ test]# touch test1.txt
[root@iZiiwad3d3m7zzZ test]# echo helloworld >>test1.txt
[root@iZiiwad3d3m7zzZ test]# ls
test1.txt
[root@iZiiwad3d3m7zzZ test]# docker ps
CONTAINER ID        IMAGE                 COMMAND                  CREATED              STATUS              PORTS                                            NAMES
51b1326faaa7        centos                "/bin/bash"              About a minute ago   Up About a minute                                                    peaceful_payne

# 进入容器并查看目录,发现数据已经同步更新
[root@iZiiwad3d3m7zzZ test]# docker exec -it 51b1326faaa7 /bin/bash
[root@51b1326faaa7 /]# ls
bin  dev  etc  home  lib  lib64  lost+found  media  mnt  opt  proc  root  run  sbin  srv  sys  tmp  usr  var
[root@51b1326faaa7 /]# cd /home
[root@51b1326faaa7 home]# ls
test1.txt
[root@51b1326faaa7 home]# cat test1.txt
helloworld
#使用docker inspect 查看刚才创建的容器

        "Mounts": [
            {
                "Type": "bind",
                "Source": "/home/test", # 主机内的地址
                "Destination": "/home",#docker容器的地址
                "Mode": "",
                "RW": true,
                "Propagation": "rprivate"
            }
        ],
# 删除容器后,重新创建一个容器挂载相同的目录，可以在容器中找到一开始创建的文件,这就达到了容器的持久化
[root@iZiiwad3d3m7zzZ test]# docker rm -f 51b1326faaa7
51b1326faaa7

# 重新创建容器,挂载相同的目录
[root@iZiiwad3d3m7zzZ test]# docker run -it -v /home/test:/home centos /bin/bash
# 在容器中能找到一开始创建的文件
[root@9eddbffd4d16 /]# cd /home
[root@9eddbffd4d16 home]# ls
test1.txt
```

**注:即使容器已经停止了,也可以进行数据同步**

**注: 挂载是主机目录下的文件 覆盖容器目录下的文件**

卷挂载的好处: 以后修改可以直接在本地修改即可,容器内会自动同步。

## 具名和匿名挂载

```
#匿名挂载 -v时只指定容器的目录
直接使用-v 容器路径
docker run -d -P --name nginx01 -v /etc/nginx nginx

# 查看匿名卷
docker volume ls
```

[![tm7k6A.th.png](https://s1.ax1x.com/2020/05/29/tm7k6A.png?ynotemdtimestamp=1618897827085)](https://imgchr.com/i/tm7k6A)

```
# 使用docker inspect 容器id ，查看mounts挂载


  "Mounts": [
            {
                "Type": "volume",
                "Name": "a70c13d31c8005c2dc1d32fffc3d06330040f3a0c4f3955d4ccb152dbc9e5628",
                "Source": "/var/lib/docker/volumes/a70c13d31c8005c2dc1d32fffc3d06330040f3a0c4f3955d4ccb152dbc9e5628/_data",
                "Destination": "/etc/nginx",
                "Driver": "local",
                "Mode": "",
                "RW": true,
                "Propagation": ""
            }
        ],


#对照docker volume ls 发现图中每一串值 都是一个文件目录

# cd 进入目录, 数据已同步
[root@iZiiwad3d3m7zzZ etc]# cd /var/lib/docker/volumes/a70c13d31c8005c2dc1d32fffc3d06330040f3a0c4f3955d4ccb152dbc9e5628/_data
[root@iZiiwad3d3m7zzZ _data]# ls
conf.d  fastcgi_params  koi-utf  koi-win  mime.types  modules  nginx.conf  scgi_params  uwsgi_params  win-utf
[root@iZiiwad3d3m7zzZ _data]#
```

**结论:所有匿名挂载都是挂载到/var/lib/docker/volumes/xxxxxx/_data下**

使用具名挂载可以方便的找到挂载的卷, 因此大多数下都是使用具名挂载

```
# 如何确定是具名挂载还是匿名挂载,还是指定目录挂载
-v 容器内路径  # 匿名挂载
-v 卷名:容器内路径 # 具名挂载
-v /宿主机路径:容器内路径#指定路径进行挂载
```

拓展

```
# 通过 指定容器内路径+ :rw/ro  
ro readonly#只读  只能通过宿主机修改
rw readwrite#可读可写 默认配置

docker run  ...  -v xxx:/etc/filename:ro imagesname
docker run  ...  -v xxx:/etc/filename:rw imagesname
```





## 基本操作

### 镜像下载

`docker pull xx`

### 镜像运行

`docker run -it --rm ubuntu:18.04 bash`

### 镜像环境保存上传(用于保存现场)

`docker commit [选项] <容器ID或容器名> [<仓库名>[:<标签>]]`

![file://C:/Users/qwrdxer/AppData/Local/Temp/.IO42A2/1.png](file://C:/Users/qwrdxer/AppData/Local/Temp/.IO42A2/1.png)





## docker file 定制镜像



### FROM 命令

```dockerfile
FROM nginx
RUN echo '<h1>Hello, Docker!</h1>' > /usr/share/nginx/html/index.html
```

指定基础镜像, 可以指定mysql 、 NGINX等，也可指定Ubuntu  centos等更为基础的镜像



其中 `FROM scratch`是一个特殊的基础镜像,如果你以 `scratch` 为基础镜像的话，意味着你不以任何镜像为基础，接下来所写的指令将作为镜像第一层开始存在。

> Docker 的 `FROM scratch` 指令是一种特殊情况，它表示你从一个空白的基础镜像开始构建你的 Docker 镜像，而不是从一个已有的基础镜像构建。这可以用于创建极小的 Docker 镜像，通常用于静态二进制文件的部署，如 Go 程序。
>
> 以下是一个简单的使用 `FROM scratch` 构建 Docker 镜像的示例：
>
> 1. 首先，创建一个包含可执行文件的 Go 程序，比如 `main.go`：
>
> ```
> goCopy codepackage main
> 
> import "fmt"
> 
> func main() {
>     fmt.Println("Hello, World!")
> }
> ```
>
> 1. 使用 Go 工具构建该程序，以生成可执行文件。在程序的目录中运行以下命令：
>
> ```
> bashCopy code
> go build -o myapp main.go
> ```
>
> 这将生成一个名为 `myapp` 的可执行文件。
>
> 1. 创建一个 Dockerfile，以使用 `FROM scratch` 构建 Docker 镜像。在与 `myapp` 可执行文件相同的目录中创建一个名为 `Dockerfile` 的文件，内容如下：
>
> ```
> DockerfileCopy code# 使用空白基础镜像
> FROM scratch
> 
> # 将可执行文件复制到镜像中
> COPY myapp /myapp
> 
> 
> # 指定容器启动时运行的命令
> CMD ["/myapp"]
> ```
>
> 这个 Dockerfile 使用了 `FROM scratch` 指令，然后将 `myapp` 可执行文件复制到镜像中，并为它设置了执行权限。最后，使用 `CMD` 指定容器启动时运行的命令，即运行 `myapp`。
>
> 1. 构建 Docker 镜像。在包含 Dockerfile 的目录中运行以下命令：
>
> ```
> bashCopy code
> docker build -t myapp-image .
> ```
>
> 这将使用 Dockerfile 构建一个名为 `myapp-image` 的 Docker 镜像。
>
> 1. 运行 Docker 容器。现在，您可以使用以下命令在容器中运行应用程序：
>
> ```
> bashCopy code
> docker run myapp-image
> ```
>
> 这将启动一个容器，并运行名为 `myapp` 的可执行文件，输出 "Hello, World!"。
>
> 这个示例演示了如何使用 `FROM scratch` 构建一个包含最小文件系统的 Docker 镜像，并将一个可执行文件添加到其中以运行应用程序。这种方法适用于需要极小镜像的特殊用例。



### RUN命令

`RUN` 指令是用来执行命令行命令的。由于命令行的强大能力，`RUN` 指令在定制镜像时是最常用的指令之一。其格式有两种：

- *shell* 格式：`RUN <命令>`，就像直接在命令行中输入的命令一样。刚才写的 Dockerfile 中的 `RUN` 指令就是这种格式。

```docker
RUN echo '<h1>Hello, Docker!</h1>' > /usr/share/nginx/html/index.html
```

- *exec* 格式：`RUN ["可执行文件", "参数1", "参数2"]`，这更像是函数调用中的格式。



使用RUN进行apt更新

```dockerfile
FROM debian:stretch

RUN set -x; buildDeps='gcc libc6-dev make wget' \
    && apt-get update \
    && apt-get install -y $buildDeps \
    && wget -O redis.tar.gz "http://download.redis.io/releases/redis-5.0.3.tar.gz" \
    && mkdir -p /usr/src/redis \
    && tar -xzf redis.tar.gz -C /usr/src/redis --strip-components=1 \
    && make -C /usr/src/redis \
    && make -C /usr/src/redis install \
    && rm -rf /var/lib/apt/lists/* \
    && rm redis.tar.gz \
    && rm -r /usr/src/redis \
    && apt-get purge -y --auto-remove $buildDeps
```

### copy命令

- `COPY [--chown=<user>:<group>] <源路径>... <目标路径>
- COPY [--chown=<user>:<group>] ["<源路径1>",... "<目标路径>"]

### add命令

`ADD` 指令和 `COPY` 的格式和性质基本一致。但是在 `COPY` 基础上增加了一些功能。

比如 `<源路径>` 可以是一个 `URL`，这种情况下，Docker 引擎会试图去下载这个链接的文件放到 `<目标路径>` 去。下载后的文件权限自动设置为 `600`，如果这并不是想要的权限，那么还需要增加额外的一层 `RUN` 进行权限调整，另外，如果下载的是个压缩包，需要解压缩，也一样还需要额外的一层 `RUN` 指令进行解压缩。所以不如直接使用 `RUN` 指令，然后使用 `wget` 或者 `curl` 工具下载，处理权限、解压缩、然后清理无用文件更合理。因此，这个功能其实并不实用，而且不推荐使用。

如果 `<源路径>` 为一个 `tar` 压缩文件的话，压缩格式为 `gzip`, `bzip2` 以及 `xz` 的情况下，`ADD` 指令将会自动解压缩这个压缩文件到 `<目标路径>` 去。

### CMD命令

`CMD` 指令的格式和 `RUN` 相似，也是两种格式：

- `shell` 格式：`CMD <命令>`
- `exec` 格式：`CMD ["可执行文件", "参数1", "参数2"...]`
- 参数列表格式：`CMD ["参数1", "参数2"...]`。在指定了 `ENTRYPOINT` 指令后，用 `CMD` 指定具体的参数。



Docker 不是虚拟机，容器就是进程。既然是进程，那么在启动容器的时候，需要指定所运行的程序及参数。CMD 指令就是用于指定默认的容器主进程的启动命令的。

在运行时可以指定新的命令来替代镜像设置中的这个默认命令，比如，ubuntu 镜像默认的 CMD 是 /bin/bash，如果我们直接 `docker run -it ubuntu` 的话，会直接进入 bash。我们也可以在运行时指定运行别的命令，如 `docker run -it ubuntu cat /etc/os-release`  输出了系统版本信息(然后进程结束)

> Docker 不是虚拟机，容器中的应用都应该以前台执行，而不是像虚拟机、物理机里面那样，用 `systemd` 去启动后台服务，容器内没有后台服务的概念。
>
> 一些初学者将 `CMD` 写为：
>
> 
>
> ```docker
> CMD service nginx start
> ```
>
> 1
>
> 然后发现容器执行后就立即退出了。甚至在容器内去使用 `systemctl` 命令结果却发现根本执行不了。这就是因为没有搞明白前台、后台的概念，没有区分容器和虚拟机的差异，依旧在以传统虚拟机的角度去理解容器。
>
> 对于容器而言，其启动程序就是容器应用进程，容器就是为了主进程而存在的，主进程退出，容器就失去了存在的意义，从而退出，其它辅助进程不是它需要关心的东西。
>
> 而使用 `service nginx start` 命令，则是希望 upstart 来以后台守护进程形式启动 `nginx` 服务。而刚才说了 `CMD service nginx start` 会被理解为 `CMD [ "sh", "-c", "service nginx start"]`，因此主进程实际上是 `sh`。那么当 `service nginx start` 命令结束后，`sh` 也就结束了，`sh` 作为主进程退出了，自然就会令容器退出。
>
> 正确的做法是直接执行 `nginx` 可执行文件，并且要求以前台形式运行。比如：
>
> 
>
> ```docker
> CMD ["nginx", "-g", "daemon off;"]
> ```



### ENTRYPOINT 入口点

当指定了 `ENTRYPOINT` 后，`CMD` 的含义就发生了改变，不再是直接的运行其命令，而是将 `CMD` 的内容作为参数传给 `ENTRYPOINT` 指令，换句话说实际执行时，将变为：

```bash
<ENTRYPOINT> "<CMD>"
```

### ENV 设置环境变量



格式有两种：

- `ENV <key> <value>`
- `ENV <key1>=<value1> <key2>=<value2>...`

这个指令很简单，就是设置环境变量而已，无论是后面的其它指令，如 `RUN`，还是运行时的应用，都可以直接使用这里定义的环境变量。



### VOLUME匿名卷

```docker
VOLUME /data
```

这里的 `/data` 目录就会在容器运行时自动挂载为匿名卷，任何向 `/data` 中写入的信息都不会记录进容器存储层，从而保证了容器存储层的无状态化。当然，运行容器时可以覆盖这个挂载设置。比如：

```bash
$ docker run -d -v mydata:/data xxxx
```





### EXPOSE 声明端口

 `EXPOSE` 仅仅是声明容器打算使用什么端口而已，并不会自动在宿主进行端口映射。



### WORKDIR 指定工作目录

格式为 `WORKDIR <工作目录路径>`。

使用 `WORKDIR` 指令可以来指定工作目录（或者称为当前目录），以后各层的当前目录就被改为指定的目录，如该目录不存在，`WORKDIR` 会帮你建立目录。

> 之前提到一些初学者常犯的错误是把 `Dockerfile` 等同于 Shell 脚本来书写，这种错误的理解还可能会导致出现下面这样的错误：
>
> 
>
> ```docker
> RUN cd /app
> RUN echo "hello" > world.txt
> ```
>
> 
>
> 如果将这个 `Dockerfile` 进行构建镜像运行后，会发现找不到 `/app/world.txt` 文件，或者其内容不是 `hello`。原因其实很简单，在 Shell 中，连续两行是同一个进程执行环境，因此前一个命令修改的内存状态，会直接影响后一个命令；而在 `Dockerfile` 中，这两行 `RUN` 命令的执行环境根本不同，是两个完全不同的容器。这就是对 `Dockerfile` 构建分层存储的概念不了解所导致的错误。

### USER指定用户名

```
USER <用户名>[:<用户组>]
```

`USER` 指令和 `WORKDIR` 相似，都是改变环境状态并影响以后的层。`WORKDIR` 是改变工作目录，`USER` 则是改变之后层的执行 `RUN`, `CMD` 以及 `ENTRYPOINT` 这类命令的身份

### HEALTHCHECK 健康检查

- `HEALTHCHECK [选项] CMD <命令>`：设置检查容器健康状况的命令
- `HEALTHCHECK NONE`：如果基础镜像有健康检查指令，使用这行可以屏蔽掉其健康检查指令

```docker
FROM nginx
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
HEALTHCHECK --interval=5s --timeout=3s \
  CMD curl -fs http://localhost/ || exit 1
```



### SHELL命令

格式：`SHELL ["executable", "parameters"]`

```
SHELL` 指令可以指定 `RUN` `ENTRYPOINT` `CMD` 指令的 shell，Linux 中默认为 `["/bin/sh", "-c"]
```



```docker
SHELL ["/bin/sh", "-c"]

RUN lll ; ls

SHELL ["/bin/sh", "-cex"]

RUN lll ; ls
```





### 构建镜像

在 `Dockerfile` 文件所在目录执行

```bash
 docker build -t nginx:v3 .
```





## dockerfile实例



`````dockerfile
FROM ubuntu:18.04
CMD  apt insall n
`````







## Docker 网络

[![tm7xjs.th.png](https://s1.ax1x.com/2020/05/29/tm7xjs.png?ynotemdtimestamp=1618897827085)](https://imgchr.com/i/tm7xjs)

安装docker使用 ip addr会发现有一个docker0网卡

```
# 问题: docker是如何处理容器网络访问的?
```

[![tmH9H0.th.png](https://s1.ax1x.com/2020/05/29/tmH9H0.png?ynotemdtimestamp=1618897827085)](https://imgchr.com/i/tmH9H0)

```
# 启动tomcat容器
docker run -d -P --name tomcat01 tomcat

#执行ip addr命令 查看网卡 信息,
 docker exec -it tomcat01 ip addr
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
130: eth0@if131: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default
    link/ether 02:42:ac:11:00:02 brd ff:ff:ff:ff:ff:ff link-netnsid 0
    inet 172.17.0.2/16 brd 172.17.255.255 scope global eth0
       valid_lft forever preferred_lft forever

# 发现容器启动后会得到一个这样的 130: eth0@if131 ip地址,这是docker分配的

# linux能不能ping通这个容器?

[root@iZiiwad3d3m7zzZ ~]# ping 172.17.0.2
PING 172.17.0.2 (172.17.0.2) 56(84) bytes of data.
64 bytes from 172.17.0.2: icmp_seq=1 ttl=64 time=0.072 ms
64 bytes from 172.17.0.2: icmp_seq=2 ttl=64 time=0.058 ms
# 可以,说明linux 可以ping通容器内部。

#主机再次执行ip addr 后,发现又多了一张网卡
131: veth6cdfe0d@if130: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue master docker0 state UP group default
    link/ether 06:39:bd:95:73:86 brd ff:ff:ff:ff:ff:ff link-netnsid 0
```

> 原理

1. 每启动一个容器,docker都会给docker容器分配一个ip,只要安装了docker,就会有一个网卡docker0, 使用的就是evth-pair技术.
2. 再次启动一个容器tomcat02,会发现又多了一个网卡
3. [![tmHGgH.th.png](https://s1.ax1x.com/2020/05/29/tmHGgH.png?ynotemdtimestamp=1618897827085)](https://imgchr.com/i/tmHGgH)

```
# 很容易发现,创建容器产生的网卡都是成对的, 如第一个容器的 130和主机的131对应

# evth-pair 就是一对的虚拟机设备接口,他们都是成对出现的,一段连接主协议,一段彼此相连

#正因为这个特性 ，evth-pair 作为一个桥梁,连通着各种虚拟网络设备的
```

1. 测试tomcat01 和tomcat02是否可以ping通

```
docker exec -it tomcat02 ping 172.17.0.3

#可以ping通
```

**结论：**

1. tomcat01和tomcat02 可以理解为使用公用的docker0 连接
2. 所有的容器在不指定网络的情况下都是docker0路由的,docker0会给容器分配一个默认的可用ip

> 思考一个场景: 若有一个服务,需要根据访问其他容器,但容器重启后ip重新分配,----> 能否通过服务名(容器名)访问容器服务?

```
$ docker exec --it tomcat02 ping tomcat01
ping: tomcat01: Name or service not known
#报错！ 如何解决？

#在创建容器的时候绑定服务的ip  使用 --link 来绑定服务名

[root@iZiiwad3d3m7zzZ ~]# docker run -d -P --name tomcat03 --link tomcat02 tomcat
01fa84c59ba1f005220abac5761ab135793fcdc895420d61a1dc996e0a3678bf
[root@iZiiwad3d3m7zzZ ~]# docker exec -it tomcat03 ping tomcat02
PING tomcat02 (172.17.0.3) 56(84) bytes of data.
64 bytes from tomcat02 (172.17.0.3): icmp_seq=1 ttl=64 time=0.102 ms

# 原理:  容器内部的hosts文件绑定了 tomcat02
[root@iZiiwad3d3m7zzZ ~]# docker exec -it tomcat03 cat /etc/hosts
127.0.0.1       localhost

172.17.0.3      tomcat02 1f56aac5004f
172.17.0.4      01fa84c59ba1
docker network ls #查看容器网卡信息

NETWORK ID          NAME                DRIVER              SCOPE
0457aba507e3        bridge              bridge              local
702f5023c1fe        host                host                local
dae9e02f4e11        none                null                local


docker inspect 网络ID #查看网络信息

        "Containers": {
            "01fa84c59ba1f005220abac5761ab135793fcdc895420d61a1dc996e0a3678bf": {
                "Name": "tomcat03",
                "EndpointID": "234c45609a5a840380c4003201099266fe6d27961ac14f2eab349d8004cbd9a9",
                "MacAddress": "02:42:ac:11:00:04",
                "IPv4Address": "172.17.0.4/16",
                "IPv6Address": ""
            },
            "1f56aac5004f9183688686d155cf444bbc3829581b92915fb8ed8dada8b428ef": {
                "Name": "tomcat02",
                "EndpointID": "0707d81268880ab6957b8551374935f28085a000279c6bae69d92e1302de4453",
                "MacAddress": "02:42:ac:11:00:03",
                "IPv4Address": "172.17.0.3/16",
                "IPv6Address": ""
            },
            "6cfd7d3ad129a00f615a35107aa034b6bb99861042462fe65947db19a188fc8f": {
                "Name": "tomcat01",
                "EndpointID": "ed8815b69ba59d276634b4113dcdf2015c168e5f5398152f1d744a141ba36de7",
                "MacAddress": "02:42:ac:11:00:02",
                "IPv4Address": "172.17.0.2/16",
                "IPv6Address": ""
            }
        },

#查看tomcat03 信息
$ docker inspect tomcat03

  "Links": [
                "/tomcat02:/tomcat03/tomcat02"
            ],
发现有连接的标志
本质探究: --link 就是在hosts配置中增加了一个 172.17.0.3      tomcat02 1f56aac5004f 使用 --link麻烦, 可以自定义网络
```

## 自定义网络

> 查看所有的docker网络

```
[root@iZiiwad3d3m7zzZ ~]# docker network ls
NETWORK ID          NAME                DRIVER              SCOPE
0457aba507e3        bridge              bridge              local
702f5023c1fe        host                host                local
dae9e02f4e11        none                null                local
```

**网络模式**

bridge: 桥接(默认)

none: 不配置网络

host: 主机模式,和宿主机共享网络

container: 容器内网络连通(用的少)

**测试**

```
# 直接启动容器 默认命令  --net bridge 这就是docker0
$ docker run -d -P --name tomcat01 tomcat
$ docker run -d -P --name tomcat01 --net bridge tomcat

#docker0 特点:默认, 无法直接域名访问, 需要使用 --link

# 创建一个自定义网络

#--dirver bridge
#--subnet 192.168.0.0/16
#--gateway 192.168.0.1

[root@iZiiwad3d3m7zzZ ~]# docker network create --driver bridge --subnet 192.168.0.0/16 --gateway 192.168.0.1 myfirstnet
52fdb3ca6d7f9e0bb0ec01df0b1726c3d75081814215c45cf762b9f018f0b132

[root@iZiiwad3d3m7zzZ ~]# docker network ls
NETWORK ID          NAME                DRIVER              SCOPE
0457aba507e3        bridge              bridge              local
702f5023c1fe        host                host                local
52fdb3ca6d7f        myfirstnet          bridge              local
dae9e02f4e11        none                null                local

# 指定容器的网络
$docker run -d -P --name tomcat-net-01 --net myfirstnet tomcat
$docker run -d -P --name tomcat-net-02 --net myfirstnet tomcat


# 测试连通

[root@iZiiwad3d3m7zzZ ~]# docker exec -it 1af8dfd43a5d ping tomcat-net-01
PING tomcat-net-01 (192.168.0.2) 56(84) bytes of data.
64 bytes from tomcat-net-01.myfirstnet (192.168.0.2): icmp_seq=1 ttl=64 time=0.065 ms

64 bytes from tomcat-net-01.myfirstnet (192.168.0.2): icmp_seq=2 ttl=64 time=0.061 ms
^C

# 发现不需要使用 --link绑定 直接就可以使用容器名ping通
```

好处:不同的集群使用不同的网络,保证集群是安全和健康的。

[![tmHyvj.th.png](https://s1.ax1x.com/2020/05/29/tmHyvj.png?ynotemdtimestamp=1618897827085)](https://imgchr.com/i/tmHyvj)

## 网络连通

[![tmHWV0.th.png](https://s1.ax1x.com/2020/05/29/tmHWV0.png?ynotemdtimestamp=1618897827085)](https://imgchr.com/i/tmHWV0)

测试:

```
#启动一个使用默认网络的容器,无法ping通使用其他网络下的容器
[root@iZiiwad3d3m7zzZ ~]# docker run -d -P --name tomcat-03 tomcat
a6c6a1fd6ec54dd9e1e38c27aba45e0b6f5c4c282408a69c575f15639e12584d

[root@iZiiwad3d3m7zzZ ~]# docker exec -it tomcat-03 ping tomcat-net-01
ping： tomcat-net-01:Name or service not known

#docker network connect : 连接一个容器到一个网络

#测试打通tomcat-03 与tomcat-net-01
[root@iZiiwad3d3m7zzZ ~]# docker network connect myfirstnet tomcat-03
[root@iZiiwad3d3m7zzZ ~]# docker network inspect myfirstnet
 "Containers": {
            "1af8dfd43a5df52ddf89d852d744e375bcaae3ba40e7c20d0ab86540fa36a2d2": {
                "Name": "tomcat-net-02",
                "EndpointID": "bb33917cb8b9d0f758ddc4b67182d067c928b191a767e571f066ea4414f36173",
                "MacAddress": "02:42:c0:a8:00:03",
                "IPv4Address": "192.168.0.3/16",
                "IPv6Address": ""
            },
            "91a3bb736a2a6b125fb577443302488bc1d76c0988c0c56ec8f51a032a43564f": {
                "Name": "tomcat-net-01",
                "EndpointID": "0b3d23e32cd45767d27a8d705d1e6e4d32b1559f147c85e956f16fcb923276ca",
                "MacAddress": "02:42:c0:a8:00:02",
                "IPv4Address": "192.168.0.2/16",
                "IPv6Address": ""
            },
            "a6c6a1fd6ec54dd9e1e38c27aba45e0b6f5c4c282408a69c575f15639e12584d": {
                "Name": "tomcat-03",
                "EndpointID": "50e3b50310d9f0684233d7f9ece3df3f88094dd8ef51441be00067b578aef6f1",
                "MacAddress": "02:42:c0:a8:00:04",
                "IPv4Address": "192.168.0.4/16",
                "IPv6Address": ""
            }
        },
#发现自定义网络添加了一个ip给 其他网络的容器

#一个容器拥有了两个ip
```

## docker常用镜像记录

**centos 安装docker**

```bash
sudo yum remove docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-engine

#安装用到的工具 
sudo yum install -y yum-utils

#添加仓库 
 sudo yum-config-manager --add-repo http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo

#加速 
yum makecache fast

#安装docker
sudo yum -y install docker-ce docker-ce-cli contanerd.io

#启动docker 
sudo systemctl start docker

#测试, 查看docker版本 
docker version


#安装指定版本docker
#列出docker版本
yum list docker-ce.x86_64 --showduplicates | sort -r
# 查看cli版本
yum list docker-ce.x86_64 docker-ce-cli.x86_64 --showduplicates | sort -r | grep 18.09
#安装
yum install -y docker-ce-18.09.9 docker-ce-cli-18.09.9 containerd.io
```

**安装docker脚本**

```bash
curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
```

**配置国内加速**

```bash
创建或修改 /etc/docker/daemon.json 文件，修改为如下形式

{
  "registry-mirrors": [
    "https://registry.docker-cn.com",
    "http://hub-mirror.c.163.com",
    "https://docker.mirrors.ustc.edu.cn"
  ]
}

$ sudo systemctl daemon-reload
$ sudo systemctl restart docker
```

**哆啦靶场docker一键配置**

/* pull 镜像 */
sudo docker pull registry.cn-hangzhou.aliyuncs.com/duolass/duola:1.2
/* 一键启动 -p 主机端口:容器端口 * 主机端口可自己指定，容器端口为80/
    sudo docker run -d -p 7777:80 registry.cn-hangzhou.aliyuncs.com/duolass/duola:1.2

**vulfocus**
#下载
docker pull vulfocus/vulfocus 
启动环境 
$docker run -d -p 80:80 -v /var/run/docker.sock:/var/run/docker.sock -e VUL_IP=xx

**webbug4.0**

docker pull area39/webug
docker run -d -P area39/webug

**metaspolitable2**
docker pull tleemcjr/metasploitable2
docker run --name msfable -it tleemcjr/metasploitable2:latest sh -c "/bin/services.sh && bash"

**dvwa**

Pull image: docker pull citizenstig/dvwa

sudo docker run -d -p 6666:80 -p 3307:3306 -e MYSQL_PASS="123456" citizenstig/dvwa

**awvs1**3

```bash
pull 拉取下载镜像

docker pull secfa/docker-awvs

将Docker的3443端口映射到物理机的 13443端口

docker run -it -d -p 13443:3443 secfa/docker-awvs

容器的相关信息

awvs13 username: admin@admin.com
awvs13 password: Admin123
AWVS版本：13.0.200217097
```

**awvs14**

```shell
#awvs14 docker
docker run -it -d --name awvs -p 3443:3443 xrsec/awvs:v14
ip:3443
awvs@awvs.com
Awvs@awvs.com
```

**wordpress**

```bash
docker run -d --privileged=true --name OLDMysql -v /data/mysql:/var/lib/mysql -e MYSQL_ROOT_PASSWORD=123456 -p 33306:3306 mysql:5.6

docker run -d --name OLDwp -e WORDPRESS_DB_HOST=mysql -e WORDPRESS_DB_USER=root -e WORDPRESS_DB_PASSWORD=123456 -e WORDPRESS_DB_NAME=myword -p 1080:80 --link OLDMysql:mysql wordpress
```

**nessus 8.10**

```bash
docker pull leishianquan/awvs-nessus:v3

 docker run -itd -p 3443:3443 -p 8834:8834 leishianquan/awvs-nessus:v3 

 进入容器

 docker exec -it [容器id] /bin/bash 

启动

 nessus /etc/init.d/nessusd start 

 ctrl+A  ctrl+D 退出 容器 

用户名 密码 leishi leishianquan
```
