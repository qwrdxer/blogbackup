---
title: The Way To Go
toc: true
categories:
  - 编程语言
  - Go
date: 2022-04-08 18:09:20
tags:
---









### Go 方法

map和struct vs new() 和make()

make 用于 slices maps channel

试图通过make创建一个结构体会报错

试图通过new创建一个map变量，需要为其初始化后才能使用。



结构体中的标签(tag)

可以通过反射获取



Go方法

1. Go 方法是作用在接收者（receiver）上的一个函数，接收者是某种类型的变量。因此方法是一种特殊类型的函数。

2. 接收者类型可以是（几乎）任何类型，不仅仅是结构体类型：任何类型都可以有方法，甚至可以是函数类型，可以是 int、bool、string 或数组的别名类型。但是接收者不能是一个接口类型，因为接口是一个抽象定义，但是方法却是具体实现；如果这样做会引发一个编译错误：**invalid receiver type…**
3. 最后接收者不能是一个指针类型(ptr)，但是它可以是任何其他允许类型的指针。
4. 定义方法格式如下

`````go
func (recv receiver_type) methodName(parameter_list) (return_value_list) { ... }
`````

思考1: 什么时候使用方法？
当需要对结构体中原始的数据进行相互之间或单独处理时，考虑使用方法；

思考2: 当接受者为值或指针时?

**由值作为接收者，被调用时，是对对象的拷贝，指针作为接收者，被调用时，是对对象本身的操作**。

https://blog.csdn.net/u014239709/article/details/89552933
https://www.cntofu.com/book/14/eBook/11.6.md



*<!-- more -->* 

### Go接口

接口定义了一组方法（方法集），但是这些方法不包含（实现）代码：它们没有被实现（它们是抽象的）。接口里也不能包含变量。

**类型不需要显式声明它实现了某个接口：接口被隐式地实现。多个类型可以实现同一个接口**。

**实现某个接口的类型（除了实现接口方法外）可以有其他的方法**。

**一个类型可以实现多个接口**。

**接口类型可以包含一个实例的引用， 该实例的类型实现了此接口（接口是动态类型）**。

即使接口在类型之后才定义，二者处于不同的包中，被单独编译：只要类型实现了接口中**所有**的方法，它就实现了此接口。



接口定义

`````go
type Namer interface {
    Method1(param_list) return_type
    Method2(param_list) return_type
    ...
}
`````

接口实现

`````go
func(xxx type1) Method1(param_list) return_type
`````



两个类型 `stockPosition` 和 `car`，它们都有一个 `getValue()` 方法，我们可以定义一个具有此方法的接口 `valuable`。接着定义一个使用 `valuable` 类型作为参数的函数 `showValue()`，所有实现了 `valuable` 接口的类型都可以用这个函数。





一个接口可以包含一个或多个其他的接口，这相当于直接将这些内嵌接口的方法列举在外层接口中一样。



通常我们可以使用 **类型断言** 来测试在某个时刻 `varI` 是否包含类型 `T` 的值：

`````go
if v, ok := varI.(T); ok {  // checked type assertion
    Process(v)
    return
}
// varI is not of type T
`````

接口变量的类型也可以使用一种特殊形式的 `switch` 来检测：**type-switch** （下面是示例 11.4 的第二部分）：

```go
switch t := areaIntf.(type) {
case *Square:
	fmt.Printf("Type Square %T with value %v\n", t, t)
case *Circle:
	fmt.Printf("Type Circle %T with value %v\n", t, t)
case nil:
	fmt.Printf("nil value: nothing to check?\n")
default:
	fmt.Printf("Unexpected type %T\n", t)
}
```

可以用 `type-switch` 进行运行时类型分析，但是在 `type-switch` 不允许有 `fallthrough` 。



假定 `v` 是一个值，然后我们想测试它是否实现了 `Stringer` 接口，可以这样做：

```go
type Stringer interface {
    String() string
}

if sv, ok := v.(Stringer); ok {
    fmt.Printf("v implements String(): %s\n", sv.String()) // note: sv, not v
}
```





接口是一种契约，实现类型必须满足它，它描述了类型的行为，规定类型可以做什么。接口彻底将类型能做什么，以及如何做分离开来，使得相同接口的变量在不同的时刻表现出不同的行为，这就是多态的本质。

接口变量中存储的具体值是不可寻址的，幸运的是，如果使用不当编译器会给出错误



在接口上调用方法时，必须有和方法定义时相同的接收者类型或者是可以从具体类型 `P` 直接可以辨识的：

- 指针方法可以通过指针调用
- 值方法可以通过值调用
- 接收者是值的方法可以通过指针调用，因为指针会首先被解引用
- 接收者是指针的方法不可以通过值调用，因为存储在接口中的值没有地址



Go 语言规范定义了接口方法集的调用规则：

- 类型 *T 的可调用方法集包含接受者为 *T 或 T 的所有方法集
- 类型 T 的可调用方法集包含接受者为 T 的所有方法
- 类型 T 的可调用方法集不包含接受者为 *T 的方法



> 稍作总结

函数: 参数为结构体、数字、字符串、数组

`````go
func func1(param) return{
	...
}
`````



方法: 作用在接收者的函数

````go
func (recv receiver_type) methodName(parameter_list) (return_value_list) { ... }
````

通过recv.methodName调用



接口函数: 参数为接口  ，用于对实现接口的成员进行管理

`````go
func func2(int interface) return{
    ...
}
`````

调用

package.func2(var)



> 空接口

**空接口或者最小接口** 不包含任何方法，它对实现不做任何要求：

```go
type Any interface {}
```

空接口类似 `Java/C#` 中所有类的基类： `Object` 类，二者的目标也很相近。

可以给一个空接口类型的变量 `var val interface {}` 赋任何类型的值。

每个空接口 变量在内存中占据两个字长：一个用来存储它包含的类型，另一个用来存储它包含的数据或者指向数据的指针。



> go官方说明
>
> https://go.dev/ref/spec



> go map中修改value的值
>
> https://blog.csdn.net/qq_37102984/article/details/121292529

一个接口的值可以赋值给另一个接口变量

https://blog.csdn.net/u010003835/article/details/51750511

### Go反射

反射是用程序检查其所拥有的结构，尤其是类型的一种能力；这是元编程的一种形式。反射可以在运行时检查类型和变量，例如它的大小、方法和 `动态` 的调用这些方法。这对于没有源代码的包尤其有用。这是一个强大的工具，除非真得有必要，否则应当避免使用或小心使用。

变量的最基本信息就是类型和值：反射包的 `Type` 用来表示一个 Go 类型，反射包的 `Value` 为 Go 值提供了反射接口。

Value 有一个 Type 方法返回 reflect.Value 的 Type。另一个是 Type 和 Value 都有 Kind 方法返回一个常量来表示类型：Uint、Float64、Slice 等等。同样 Value 有叫做 Int 和 Float 的方法可以获取存储在内部的值（跟 int64 和 float64 一样）

``````go
// blog: Laws of Reflection
package main

import (
	"fmt"
	"reflect"
)

func main() {
	var x float64 = 3.4
	fmt.Println("type:", reflect.TypeOf(x))
	v := reflect.ValueOf(x)
	fmt.Println("value:", v)
	fmt.Println("type:", v.Type())
	fmt.Println("kind:", v.Kind())
	fmt.Println("value:", v.Float())
	fmt.Println(v.Interface())
	fmt.Printf("value is %5.2e\n", v.Interface())
	y := v.Interface().(float64)
	fmt.Println(y)
}

``````

> 通过反射修改值

当 `v := reflect.ValueOf(x)` 函数通过传递一个 x 拷贝创建了 v，那么 v 的改变并不能更改原始的 x。要想 v 的更改能作用到 x，那就必须传递 x 的地址 `v = reflect.ValueOf(&x)`。

当然，此时的v仍无法进行修改，需调用v.Elem()函数，

````go
func main() {
	var x float64 = 3.4
	v := reflect.ValueOf(x)
	// setting a value:
	// v.SetFloat(3.1415) // Error: will panic: reflect.Value.SetFloat using unaddressable value
	fmt.Println("settability of v:", v.CanSet())
	v = reflect.ValueOf(&x) // Note: take the address of x.
	fmt.Println("type of v:", v.Type())
	fmt.Println("settability of v:", v.CanSet())
	v = v.Elem()
	fmt.Println("The Elem of v is: ", v)
	fmt.Println("type of v:", v.Type())
	fmt.Println("settability of v:", v.CanSet())
	v.SetFloat(3.1415) // this works!
	fmt.Println(v.Interface())
	fmt.Println(v)
}
````



> 反射结构体

有些时候需要反射一个结构类型。`NumField()` 方法返回结构内的字段数量；通过一个 for 循环用索引取得每个字段的值 `Field(i)`。

我们同样能够调用签名在结构上的方法，例如，使用索引 n 来调用：`Method(n).Call(nil)`。

`````go
package main

import (
	"fmt"
	"reflect"
)

type NotknownType struct {
	s1, s2, s3 string
}

func (n NotknownType) String() string {
	return n.s1 + " - " + n.s2 + " - " + n.s3
}

// variable to investigate:
var secret interface{} = NotknownType{"Ada", "Go", "Oberon"}

func main() {
	value := reflect.ValueOf(secret) // <main.NotknownType Value>
	typ := reflect.TypeOf(secret)    // main.NotknownType
	// alternative:
	//typ := value.Type()  // main.NotknownType
	fmt.Println(typ)
	knd := value.Kind() // struct
	fmt.Println(knd)

	// iterate through the fields of the struct:
	for i := 0; i < value.NumField(); i++ {
		fmt.Printf("Field %d: %v\n", i, value.Field(i))
		// error: panic: reflect.Value.SetString using value obtained using unexported field
		//value.Field(i).SetString("C#")
	}

	// call the first method, which is String():
	results := value.Method(0).Call(nil)
	fmt.Println(results) // [Ada - Go - Oberon]
}
`````



向反射方法中传入参数

````go

	params := make([]reflect.Value, 1)
	params[0] = reflect.ValueOf(20)
	results := value.Method(0).Call(params)
````



设置结构体成员的值

（成员的首字母应该大写，否则无法访问）

`````go
package main

import (
	"fmt"
	"reflect"
)

type T struct {
	A int
	B string
}

func main() {
	t := T{23, "skidoo"}
	s := reflect.ValueOf(&t).Elem()
	typeOfT := s.Type()
	for i := 0; i < s.NumField(); i++ {
		f := s.Field(i)
		fmt.Printf("%d: %s %s = %v\n", i,
			typeOfT.Field(i).Name, f.Type(), f.Interface())
	}
	s.Field(0).SetInt(77)
	s.Field(1).SetString("Sunset Strip")
	fmt.Println("t is now", t)
}
`````

### 接口和动态类型



接收一个（或多个）接口类型作为参数的函数，其实参可以是任何实现了该接口的类型。 `实现了某个接口的类型可以被传给任何以此接口为参数的函数` 。



> 动态方法调用





### 错误处理

Go 检查和报告错误条件的惯有方式：

- 产生错误的函数会返回两个变量，一个值和一个错误码；如果后者是 nil 就是成功，非 nil 就是发生了错误。
- 为了防止发生错误时正在执行的函数（如果有必要的话甚至会是整个程序）被中止，在调用函数后必须检查错误。

下面这段来自 pack1 包的代码 Func1 测试了它的返回值：



```go
if value, err := pack1.Func1(param1); err != nil {
	fmt.Printf("Error %s in pack1.Func1 with parameter %v", err.Error(), param1)
	return    // or: return err
} else {
	// Process(value)
}


```

### 从 panic 中恢复（Recover）

正如名字一样，这个（recover）内建函数被用于从 panic 或 错误场景中恢复：让程序可以从 panicking 重新获得控制权，停止终止过程进而恢复正常执行。

`recover` 只能在 defer 修饰的函数（参见 [6.4 节](https://www.cntofu.com/book/14/eBook/06.4.md)）中使用：用于取得 panic 调用中传递过来的错误值，如果是正常执行，调用 `recover` 会返回 nil，且没有其它效果。

总结：panic 会导致栈被展开直到 defer 修饰的 recover() 被调用或者程序中止。

下面例子中的 protect 函数调用函数参数 g 来保护调用者防止从 g 中抛出的运行时 panic，并展示 panic 中的信息：

```go
func protect(g func()) {
	defer func() {
		log.Println("done")
		// Println executes normally even if there is a panic
		if err := recover(); err != nil {
		log.Printf("run time panic: %v", err)
		}
	}()
	log.Println("start")
	g() //   possible runtime-error
}
```

> 参考文章
>
> https://www.cntofu.com/book/14/index.html