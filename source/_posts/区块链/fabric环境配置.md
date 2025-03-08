---
title: fabric环境配置
categories:
  - 区块链
date: 2024-01-08 18:47:43
tags:
---



使用Ubuntu18虚拟机进行环境搭建，参考官方文档:

# 一、 一些安装前的准备:

首先是更新和安装一些包:

```bash
apt update
apt upgrade
apt install -y jq proxychains4 git 
```

jq是一个json格式化工具 

git 下载fabric的相关github项目

国内下载太慢,使用Proxychains4代理进行下载,proxychains4 配置文件在`/etc/proxychains4.conf`

编辑配置文件如下:(注释最后一行，补上自己的socks代理)

![image-20240108185729958](fabric%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE/image-20240108185729958.png)

# 二、开始安装：

为了方便操作，所有软件都以root用户进行安装

为了提速，部分命令前增加了proxychains4

## 安装go

go的版本最好是新一点的，这里选择的是1.18

```go
proxychains4 wget https://dl.google.com/go/go1.18.6.linux-amd64.tar.gz

sudo tar -xzf go1.18.6.linux-amd64.tar.gz -C /usr/local 

sudo ln -s /usr/local/go/bin/* /usr/bin/   
```

![image-20240108190328390](fabric%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE/image-20240108190328390.png)

## 安装Docker-copmose

```bash
sudo proxychains4 curl -SL https://github.com/docker/compose/releases/download/v2.1.1/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

![image-20240108190630039](fabric%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE/image-20240108190630039.png)

## 安装fabric 和fabric samples

https://hyperledger-fabric.readthedocs.io/en/latest/install.html

```
mkdir -p $HOME/go/src/github.com/<your_github_userid>
cd $HOME/go/src/github.com/<your_github_userid>
```

将<> 中的内容替换为自己自己的GitHub ID

```bash
mkdir -p $HOME/go/src/github.com/qwrdxer
cd $HOME/go/src/github.com/qwrdxer
```

获取安装脚本

```bash
proxychains4 curl -sSLO https://raw.githubusercontent.com/hyperledger/fabric/main/scripts/install-fabric.sh chmod +x install-fabric.sh
```

运行脚本,安装fabric( 一些文件 和docker 镜像)

```bash
./install-fabric.sh
./install-fabric.sh --fabric-version 2.5.5 binary
```

![image-20240108192034593](fabric%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE/image-20240108192034593.png)

## 运行一个测试网络

https://hyperledger-fabric.readthedocs.io/en/latest/test_network.html

```bash
 #切换文件夹
cd fabric-samples/test-network
#启动测试网络
./network.sh up 
```

![image-20240108192617515](fabric%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE/image-20240108192617515.png)

创建channel

```bash
./network.sh up createChannel
```

![image-20240108204654700](fabric%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE/image-20240108204654700.png)

接下来就是chaincode ,用例中是go语言的chaincode ,需要对其进行编译，首先配置好go环境

```bash
 echo "export GO111MODULE=on" >> ~/.profile

 echo "export GOPROXY=https://goproxy.cn" >> ~/.profile

 source ~/.profile
export GO111MODULE=on
export GOPROXY=https://goproxy.io,direct
```

部署chaincode

```bash
./network.sh deployCC -ccn basic -ccp ../asset-transfer-basic/chaincode-go -ccl go
```



设置环境变量

```
export PATH=${PWD}/../bin:$PATH
export FABRIC_CFG_PATH=$PWD/../config/
export CORE_PEER_TLS_ENABLED=true
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=localhost:7051
```

执行命令

```bash
peer chaincode invoke -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --tls --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" -C mychannel -n basic --peerAddresses localhost:7051 --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" --peerAddresses localhost:9051 --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt" -c '{"function":"InitLedger","Args":[]}'
```

```bash
peer chaincode query -C mychannel -n basic -c '{"Args":["GetAllAssets"]}'
```



显示如下则是成功

![image-20240108205132653](fabric%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE/image-20240108205132653.png)





