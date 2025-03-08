---
title: C++入门
toc: true
categories:
  - 编程语言
  - C++
date: 2023-01-10 11:30:50
tags:
---



### 一个简单的实例

```C++
#include <iostream> 
using namespace std; //
int main()
{
    cout << "Hello, world!" << endl;
    return 0;
}
```

> #include <iostream>  //当要使用其他已经编写好的函数的时候，使用include来包含头文件，头文件中包含已经编写好的函数声明,并且该头文件中的函数在其他地方被实现了，可以直接使用。
>
> using namespace std; // 命名空间，同名的变量可以在不同的命名空间中，这样就不会造成各种库重复命名的冲突。std 是系统标准的命名空间，为了和用户定义的名字不重复，所以它声明在 std 这个命名空间中。另外，这个空间也像一个大包一样，包括了系统所有的支持。
>
> 如在命名空间xx和yy **xx::a** 和 **yy::a** 虽然都叫 a，但是不是同一个变量。
>
> main是一个程序的主函数，是程序开始执行的地方。

*<!-- more -->* 







> 参考文章
>
> https://www.runoob.com/cplusplus/cpp-tutorial.html