---
title: hexo相关
date: 2021-04-16 12:12:14
tags:
- 工具使用
- 脚本
categories:
- 开发工具
---

### 常用命令

```cmd
#生成静态文件并重新部署
hexo g -d

# 本地测试 访问http://localhost:4000/
hexo s

#更换主题时运行一次
hexo clean

# 创建一篇文章，”“内填写文章的名字
hexo new "blogname"

# 指定目录创建文件
hexo new "titlename" -p "path\titlename"
```

*<!-- more -->* 

### 文章相关

> 使用hexo new "blog" 命令会在_post文件夹下生成blog.md,在头部有如下配置
> 
> ![image-20210416123314164](hexo%E5%91%BD%E4%BB%A4/image-20210416123314164.png)
> 
> 可以增加自定义设置

`tags示例`

> ![image-20210416123455874](hexo%E5%91%BD%E4%BB%A4/image-20210416123455874.png)
> 
> 可以为文章添加多个tags

`categories示例`

> ![](hexo%E5%91%BD%E4%BB%A4/image-20210416123611813.png)
> 
> 可将文章归为多个类别

### 主题更换

https://hexo.io/themes/

![image-20210416122102370](hexo%E5%91%BD%E4%BB%A4/image-20210416122102370.png)

点击图片可预览，点击下面的蓝色字体进入github页面

本机进入`themes`文件夹，git下载即可, 如

git clone https://github.com/mulder21c/hexo-theme-amorfati.git

![image-20210416122304877](hexo%E5%91%BD%E4%BB%A4/image-20210416122304877.png)

记住文件名，编辑根目录下的`_config.yml`文件，更新theme的值为对应的文件夹名即可(注意空格)

![image-20210416122430800](hexo%E5%91%BD%E4%BB%A4/image-20210416122430800.png)

输入命令 `hexo g -d` 即可更换完成

若主题仍未更换，可能是浏览器换成的问题。



### 本地图片上传

https://blog.csdn.net/ayuayue/article/details/109198493

设置post_asset_folder  为 true, 安装插件 asset-image
npm install https://github.com/CodeFalling/hexo-asset-image
typora 设置图片为本地上传、优先使用相对路径
hexo clean && hexo generate && hexo s 运行查看



### 博客备份

```bash
 #插件安装
 npm install hexo-git-backup --save
 #插件更新
  npm remove hexo-git-backup
 npm install hexo-git-backup --save
```

编辑`_config.yml`

```yaml
 # 博客备份
 backup:
   type: git         # 默认不变
   theme: shoka      # 主题名称
   message: this is my  blog backup  # 提交信息
   repo:
     githu: git@github.com:0000rookie/backup-blog.git,main
     gitee: git@gitee.com:Lilbai518/backup-blog.git,master
   # coding: git@e.coding.net:lilbai518/hikki/blog-backup.git,master
   # 托管平台: 仓库地址,分支名
```



备份: hexo backup






### 参考博客

**hexo添加图片**

https://blog.csdn.net/u010996565/article/details/89196612

**hexo文章分类**

https://blog.csdn.net/maosidiaoxian/article/details/85220394

**hexo 自动文章分类**

https://blog.eson.org/pub/e2f6e239/

**备份**

https://zhuanlan.zhihu.com/p/606723790

**国内git无法正常使用**

https://www.cnblogs.com/e1sewhere/p/16574118.html
