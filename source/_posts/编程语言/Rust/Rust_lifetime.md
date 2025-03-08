---
title: Rust的生命周期
categories:
  - 编程语言
  - Rust
date: 2023-09-28 11:33:04
tags:
---

# 文章概述

1. 首先会介绍Rust的**所有权特性**，受这个所有权特性的限制,我们很多函数无法借助基本的类型来实现，因此需要引用这一类型。

2. 引用是对目标值的借用，通过引用我们可以访问甚至修改目标值,当然，引用并**没有目标值的所有权**,从直觉上来说，如果目标值已经不存在了，那引用也理所当然不存在了,所以引用受到了目标值的生命周期的限制，更明确的说，引用的生命周期必须小于(或等于)目标值的生命周期。
3. 然后我们就要详细看看生命周期了，如果代码中没有引用，那生命周期也很好推断，正是因为引用的存在，我们(和编译器)必须明确的关注**引用的生命周期**。
4. 假如我们的代码没有任何的函数调用，其值的所有创建、使用都是在main函数中实现，那编译器可以很轻松的推断出每个引用的生命周期是否合理，当然这种代码的可读性可拓展性都很非常的差，因此将部分代码抽象成函数是必要的，但函数的调用位置是很难预测的,当参数、返回值中有引用的出现，编译器就更难推断出其生命周期，因此我们需要手动的去**标注生命周期**。
5. 通过对函数中引用的生命周期进行标注，我们可以对返回值加以限制，确保这个返回值能在后续的代码中正常使用。

# 1.Rust的一些概念

## 1.1 Rust的所有权

在Rust中，

- 每个值都有一个拥有它的变量，这个变量被称为所有者。
- 一个值同时只能有一个所有者。
- 当所有者超出作用域时，值将被自动释放（回收内存）。

## 1.2 所有权的移动

在Rust中对**大多数类型**来说，变量赋值、将其传递给函数、从函数中返回这些操作都不会复制值，而是会**移动**值。

所有权转移后

```rust
    let name ="xiaoming".to_string(); //创建字符串 "xiaoming" 此时拥有它的变量为name
    let name_1=name; // 将name变量的值赋给 name_1 ,此时字符串"xiaoming"的所有权已经移交给name_1
    println!("name{}",name);//这里编译会报错,因
```



## 1.3 引用(Reference)

考虑如下代码

```rust
use std::collections::HashMap;
type Table = HashMap<String, Vec<String>>;

//我们定义一个函数用来打印Table
fn show(table: Table) {
for (artist, works) in table {
println!("works by {}:", artist);
for work in works {
println!(" {}", work);
		}
	}
}

//我们创建一个table,随后调用自定义函数将其打印出来
fn main() {
    let mut table = Table::new();
    table.insert("Gesualdo".to_string(),
        vec!["many madrigals".to_string(),
        	"Tenebrae Responsoria".to_string()]);
    table.insert("Caravaggio".to_string(),vec!["The Musicians".to_string(),
        "The Calling of St. Matthew".to_string()]);
    table.insert("Cellini".to_string(),
        vec!["Perseus with the head of Medusa".to_string(),
        	"a salt cellar".to_string()]);
    show(table);
    //如果取消下行的注释,就会编译失败,因为table已经被show函数消耗掉了
    //assert_eq!(table["Gesualdo"][0], "many madrigals"); 
}
```

总之,show函数获得了table的所有权，随后的for循环完全消耗掉了table这个变量, 因此调用下面的宏就会报错，因为整个table已经消耗掉了

```rust
assert_eq!(table["Gesualdo"][0], "many madrigals"); 
```

我们自定义的show函数的原本意图是打印出表格的内容但不会影响传入的参数，受Rust 的所有权特性影响显然无法达到预期的效果。



我们可以使用引用(Reference)来达到这一目的,引用是一种非拥有型指针,任何引用的**生命周期**都不可能超出它指向的那个值。为了强调这一点,Rust把创建对某个值的引用称为借用(borrow)那个值。

修改代码如下,即可实现我们预期的功能。

```rust
//show中的table类型为共享引用，函数只能访问而不能修改值
fn show(table: &Table) {
    for (artist, works) in table {
    	println!("works by {}:", artist);
        for work in works {
        	println!(" {}", work);
        }
    }
}
```

主函数中这样调用show函数,接收参数的引用,这样我们的输出目的就达到了

```rust
show(&table);
```

总之,通过引用我们可以在不拥有一个值的情况下去访问那个值, 当然，因为引用没有目标值的所有权，那它不可避免的就要受目标值的生命周期的限制。



# 2.引用的生命周期

在《Rust程序设计语言》中更详尽的解释了引用的生命周期

https://rustwiki.org/zh-CN/book/ch10-03-lifetime-syntax.html



我们这里主要讨论的就是引用的生命周期，因为它有时候很难确定。

Rust每一个引用都有其生命周期,也就是引用保持有效的作用域，引用的生命周期**必须小于等于**目标值的生命周期，通过这个限制我们可以安全的访问引用的目标值



考虑如下代码

```rust
{
    let r;

    {
        let x = 5;
        r = &x;
    }

    println!("r: {}", r);
}
```



外部作用域声明了一个没有初值的变量 `r`，而内部作用域声明了一个初值为 5 的变量 `x`。在内部作用域中，我们尝试将 `r` 的值设置为一个 `x` 的引用。接着在内部作用域结束后，尝试打印出 `r` 的值。这段代码不能编译因为 `r` 引用的值在尝试使用之前就离开了作用域。

运行报错

```rust
   |
20 |             let x = 5;
   |                 - `x`变量在此处声明
21 |             r = &x;
   |                 ^^ r借用了x的值,但x的生命周期小于r的生命周期:
22 |         }
   |         - `x` 生命周期结束了,但其值仍被r借用
23 |
24 |         println!("r: {}", r);
   |                           - 这里r被使用了
```



让我们来详细的看看r和x的生命周期

```rust
{
    let r;                // ---------+-- 'a
                          //          |
    {                     //          |
        let x = 5;        // -+-- 'b  |
        r = &x;           //  |       |
    }                     // -+       |
                          //          |
    println!("r: {}", r); //          |
}                         // ---------+
```

这里将 `r` 的生命周期标记为 `'a` 并将 `x` 的生命周期标记为 `'b`。如你所见，内部的 `'b` 块要比外部的生命周期 `'a` 小得多。在编译时，Rust 比较这两个生命周期的大小，并发现 `r` 拥有生命周期 `'a`，不过它引用了一个拥有生命周期 `'b` 的对象。程序被拒绝编译，因为生命周期 `'b` 比生命周期 `'a` 要小：被引用的对象比它的引用者存在的时间更短。



一个可以正常通过编译的例子

```rust
{
    let x = 5;            // ----------+-- 'b
                          //           |
    let r = &x;           // --+-- 'a  |
                          //   |       |
    println!("r: {}", r); //   |       |
                          // --+       |
}                         // ----------+
```



# 3.函数中的生命周期

单一的代码块中,我们或者编译器都能很轻易的推断出引用的生命周期是否满足条件,但是当我们调用一个函数,且这个函数的参数、返回值都有引用的出现，那编译器就很推测出各种引用的生命周期了。

考虑如下不能通过编译的代码，longest会比较两个引用的长度，返回较长的那一个:

```rust
fn main() {
    let string1 = String::from("abcd");
    let string2 = "xyz";

    let result = longest(string1.as_str(), string2);
    println!("The longest string is {}", result);//这样感觉上是没什么问题的,因为result指向的是String1
}

fn longest(x: &str, y: &str) -> &str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}
```

编译报错如下,编译器说这个函数返回了一个引用，但它并不清楚这个引用来自x 还是y。
```shell
error[E0106]: missing lifetime specifier
 --> src\main.rs:9:33
  |
9 | fn longest(x: &str, y: &str) -> &str {
  |               ----     ----     ^ expected named lifetime parameter
  |
  = help: this function's return type contains a borrowed value, but the signature does not say whether it is borrowed from `x` or `y`
help: consider introducing a named lifetime parameter
  |
9 | fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
  |           ++++     ++          ++          ++

```

为什么编译器要在意函数返回引用的来源呢 , 考虑如下代码:

```rust
fn main() {
    let string1 = String::from("abcd");
    let string2 = "xyz";

    let result =  longest(string1.as_str(), string2); //result指向了string1
    println!("{}",result); // result的借用发生了
    //我们增加点代码
    let string3 =string1; //string1所有权移交给了string3
    println!("{}",result);//result的借用再次发生了，报错
}

```

longest函数的返回值是其两个参数中的一个，尽管在调用longest这个函数时，两个引用&string1、&string2都是合理的引用，因此返回值&string1给result是合理的。

但是如果在后续的代码中，我们将string1 的所有权转移(string1生命周期到此为止)，并再次调用了result(引用生命周期大于目标值!),这显然就报错了。

总之，函数若返回值是参数的引用，如果不加上对引用和返回值生命周期的限制,就无法确保这个返回值是否在后续代码中能够正常使用。

为了修复这个错误，我们将增加**泛型生命周期参数**来定义引用间的关系以便借用检查器可以进行分析。

## 3.1 生命周期标注语法

在上面的报错中, 编译器给出了一个参考的bug修复方法

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str
```

其中 'a就是生命周期标注  ,生命周期参数名称必须以撇号（`'`）开头，其名称通常全是小写 。`'a` 是大多数人默认使用的名称。生命周期参数标注位于引用的 `&` 之后，并有一个空格来将引用类型与生命周期标注分隔开。

下面是三个例子

```rust
&i32        // 引用
&'a i32     // 带有显式生命周期的引用
&'a mut i32 // 带有显式生命周期的可变引用
```

生命周期标注并不改变任何引用的生命周期的长短，单个生命周期标注本身并没有多少意义,我们更在意的是多个引用的生命周期之间的关系,这种关系更多是在函数的参数和返回值上的。

## 3.2函数中的生命周期标注

与当函数签名中指定了**泛型类型**参数后就可以接受任何类型一样，当指定了**泛型生命周期**后函数也能接受任何生命周期的引用。

```rust
fn test<T> ...//test函数可以接受任何类型
fn longest<'a>...  //longest函数可以接受任何生命周期为'a的参数 
```

当我们指定了函数的泛型生命周期后,就可以使用它 了

修改后的函数如下,现在函数签名表明对于某些生命周期 `'a`，函数会获取两个参数，他们都是与生命周期 `'a` 存在的一样长的字符串 slice。函数会返回一个同样也与生命周期 `'a` 存在的一样长的字符串 slice。它的实际含义是 `longest` 函数返回的引用的生命周期与传入该函数的引用的生命周期的较小者一致。

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}
```

当具体的引用被传递给 `longest` 时，被 `'a` 所替代的具体生命周期是 `x` 的作用域与 `y` 的作用域相重叠的那一部分。换一种说法就是泛型生命周期 `'a` 的具体生命周期等同于 `x` 和 `y` 的生命周期中**较小**的那一个。因为我们用相同的生命周期参数 `'a` 标注了返回的引用值，所以返回的引用值就能保证在 `x` 和 `y` 中较短的那个生命周期结束之前保持有效。



# 4.总结

再次回顾一下本文:

1. 首先会介绍Rust的**所有权特性**，受这个所有权特性的限制,我们很多函数无法借助基本的类型来实现，因此需要引用这一类型。

2. 引用是对目标值的借用，通过引用我们可以访问甚至修改目标值,当然，引用并**没有目标值的所有权**,从直觉上来说，如果目标值已经不存在了，那引用也理所当然不存在了,所以引用受到了目标值的生命周期的限制，更明确的说，引用的生命周期必须小于(或等于)目标值的生命周期。
3. 然后我们就要详细看看生命周期了，如果代码中没有引用，那生命周期也很好推断，正是因为引用的存在，我们(和编译器)必须明确的关注**引用的生命周期**。
4. 假如我们的代码没有任何的函数调用，其值的所有创建、使用都是在main函数中实现，那编译器可以很轻松的推断出每个引用的生命周期是否合理，当然这种代码的可读性可拓展性都很非常的差，因此将部分代码抽象成函数是必要的，但函数的调用位置是很难预测的,当参数、返回值中有引用的出现，编译器就更难推断出其生命周期，因此我们需要手动的去**标注生命周期**。
5. 通过对函数中引用的生命周期进行标注，我们可以对返回值加以限制，确保这个返回值能在后续的代码中正常使用。



----

参考文章:

《Rust 程序设计语言》

《Programming Rust》

