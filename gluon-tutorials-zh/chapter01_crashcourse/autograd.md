# 使用autograd来自动求导

在机器学习中，我们通常使用**梯度下降**来更新模型参数从而求解。损失函数关于模型参数的梯度指向一个可以降低损失函数值的方向，我们不断地沿着梯度的方向更新模型从而最小化损失函数。虽然梯度计算比较直观，但对于复杂的模型，例如多达数十层的神经网络，手动计算梯度非常困难。

为此MXNet提供autograd包来自动化求导过程。虽然大部分的深度学习框架要求编译计算图来自动求导，`mxnet.autograd`可以对正常的命令式程序进行求导，它每次在后端实时创建计算图从而可以立即得到梯度的计算方法。

下面让我们一步步介绍这个包。我们先导入`autograd`。

```{.python .input  n=2}
from mxnet import autograd as ag
from mxnet import ndarray as nd
```

## 为变量附上梯度

假设我们想对函数`f = 2 * (x ** 2)`求关于`x`的导数。我们先创建变量`x`，并赋初值。

```{.python .input  n=5}
x = nd.random_normal(0, 1, shape=(3, 4))
f = 2 * (x * x)
```

当进行求导的时候，我们需要一个地方来存`x`的导数，这个可以通过NDArray的方法`attach_grad()`来要求系统申请对应的空间。

```{.python .input  n=6}
x.attach_grad()
```

下面定义`f`。**默认条件下，MXNet不会自动记录和构建用于求导的计算图**，我们需要使用autograd里的`record()`函数来显式的要求MXNet记录我们需要求导的程序。

```{.python .input  n=9}
with ag.record():
    z = 2 * (x * x)
```

接下来我们可以通过`z.backward()`来进行求导。如果`z`不是一个标量，那么`z.backward()`等价于`nd.sum(z).backward()`.

```{.python .input  n=10}
z.backward()
```

现在我们来看求出来的导数是不是正确的。注意到`y = x * 2`和`z = x * y`，所以`z`等价于`2 * x * x`。它的导数那么就是`dz/dx = 4 * x`。

```{.python .input  n=13}
print("x.grad: {0}, 4*x : {1}, is_equal: {2}".format(x.grad, 4 * x, x.grad == 4 * x))
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "x.grad: \n[[ -2.18377829  -7.08411598  -9.42251873  -1.80553794]\n [  2.16576099   2.31753445  10.7140255   -7.42432785]\n [  5.01853752  -7.90751839  -2.19509625  -0.83207703]]\n<NDArray 3x4 @cpu(0)>, 4*x : \n[[ -2.18377829  -7.08411598  -9.42251873  -1.80553794]\n [  2.16576099   2.31753445  10.7140255   -7.42432785]\n [  5.01853752  -7.90751839  -2.19509625  -0.83207703]]\n<NDArray 3x4 @cpu(0)>, is_equal: \n[[ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]]\n<NDArray 3x4 @cpu(0)>\n"
 }
]
```

## 对控制流求导

--- **注意其中的nd.norm(x)函数用法：求解X矩阵的L2范数（元素平方和再开方）**

命令式的编程的一个便利之处是几乎可以对任意的可导程序进行求导，即使里面包含了Python的控制流。考虑下面程序，里面包含控制流`for`和`if`，但循环迭代的次数和判断语句的执行都是取决于输入的值。不同的输入会导致这个程序的执行不一样。（对于计算图框架来说，这个对应于动态图，就是图的结构会根据输入数据不同而改变）。

```{.python .input  n=25}
def f(a):
    b = a * 2
    # nd.norm 用户求解矩阵的L2范数：元素平方和开方
    print (nd.norm(b).asscalar())
    while nd.norm(b).asscalar() < 1000:
        b = b * 2
    if nd.sum(b).asscalar() > 0:
        c = b
    else:
        c = 100 * b
    return c
    
a = nd.random_normal(0, 1, shape = (3, 4))
f(a)
print(a, nd.norm(a))
b = nd.array([3, 4])        
print(b)
nd.norm(b)
```

```{.json .output n=25}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "7.98367\n\n[[ 0.49383941  0.77932841 -0.90434265 -1.01030731]\n [-1.21407938 -0.39157307  2.1564064   1.31661868]\n [ 1.09382224 -0.43292627  1.82714331  0.71535987]]\n<NDArray 3x4 @cpu(0)> \n[ 3.99183464]\n<NDArray 1 @cpu(0)>\n\n[ 3.  4.]\n<NDArray 2 @cpu(0)>\n"
 },
 {
  "data": {
   "text/plain": "\n[ 5.]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 25,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们可以跟之前一样使用`record`记录和`backward`求导。

```{.python .input  n=28}
a = nd.random_normal(0, 1, shape=(3, 5))
a.attach_grad()
with ag.record():
    c = f(a)
c.backward()
print(a.grad)
```

```{.json .output n=28}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "11.4969\n\n[[ 256.  256.  256.  256.  256.]\n [ 256.  256.  256.  256.  256.]\n [ 256.  256.  256.  256.  256.]]\n<NDArray 3x5 @cpu(0)>\n"
 }
]
```

注意到给定输入`a`，其输出`f(a)=xa`，`x`的值取决于输入`a`。所以有`df/da = x`，我们可以很简单地评估自动求导的导数：

```{.python .input  n=29}
a.grad == c/a
```

```{.json .output n=29}
[
 {
  "data": {
   "text/plain": "\n[[ 1.  1.  1.  1.  1.]\n [ 1.  1.  1.  1.  1.]\n [ 1.  1.  1.  1.  1.]]\n<NDArray 3x5 @cpu(0)>"
  },
  "execution_count": 29,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 头梯度和链式法则

*注意：读者可以跳过这一小节，不会影响阅读之后的章节*

当我们在一个`NDArray`上调用`backward`方法时，例如`y.backward()`，此处`y`是一个关于`x`的函数，我们将求得`y`关于`x`的导数。数学家们会把这个求导写成`dy(x)/dx`。还有些更复杂的情况，比如`z`是关于`y`的函数，且`y`是关于`x`的函数，我们想对`z`关于`x`求导，也就是求`dz(y(x))/dx`的结果。回想一下链式法则，我们可以得到`dz(y(x))/dx = [dz(y)/dy] * [dy(x)/dx]`。当`y`是一个更大的`z`函数的一部分，并且我们希望求得`dz/dx`保存在`x.grad`中时，我们可以传入*头梯度*`dz/dy`的值作为`backward()`方法的输入参数，系统会自动应用链式法则进行计算。这个参数的默认值是`nd.ones_like(y)`。关于链式法则的详细解释，请参阅[Wikipedia](https://en.wikipedia.org/wiki/Chain_rule)。

```{.python .input  n=64}
w = nd.random_normal(0, 1, shape=(4, 5))
print("w: ", w)
w.attach_grad()
with ag.record():
    t = w * 2
t.backward()
print("w.grad:", w.grad)

mid_grad = w.grad
print("mid_grad: ", mid_grad)
t.attach_grad()

with ag.record():
    q = t * w
    
q.backward()
dq_dt = t.grad
print("dq/dt: ", dq_dt)

with ag.record():
    t = w * 2 
    q = t * w

q.backward(dq_dt)
print(w.grad)
# print(t.grad())

# q.backward(t.grad)
```

```{.json .output n=64}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "w:  \n[[ 0.1445754  -1.16423941 -0.44268459  0.72586095 -0.28939304]\n [ 0.08320015  1.06716168 -1.01652586 -0.22410059 -0.01811522]\n [-0.06476638 -0.24186628  1.3266871   2.52263689  0.65253955]\n [-0.21464504 -0.88811821  0.00586208 -0.30177689 -1.11130965]]\n<NDArray 4x5 @cpu(0)>\nw.grad: \n[[ 2.  2.  2.  2.  2.]\n [ 2.  2.  2.  2.  2.]\n [ 2.  2.  2.  2.  2.]\n [ 2.  2.  2.  2.  2.]]\n<NDArray 4x5 @cpu(0)>\nmid_grad:  \n[[ 2.  2.  2.  2.  2.]\n [ 2.  2.  2.  2.  2.]\n [ 2.  2.  2.  2.  2.]\n [ 2.  2.  2.  2.  2.]]\n<NDArray 4x5 @cpu(0)>\ndq/dt:  \n[[ 0.1445754  -1.16423941 -0.44268459  0.72586095 -0.28939304]\n [ 0.08320015  1.06716168 -1.01652586 -0.22410059 -0.01811522]\n [-0.06476638 -0.24186628  1.3266871   2.52263689  0.65253955]\n [-0.21464504 -0.88811821  0.00586208 -0.30177689 -1.11130965]]\n<NDArray 4x5 @cpu(0)>\n\n[[  8.36081877e-02   5.42181349e+00   7.83878565e-01   2.10749650e+00\n    3.34993333e-01]\n [  2.76890602e-02   4.55533600e+00   4.13329935e+00   2.00884297e-01\n    1.31264434e-03]\n [  1.67787336e-02   2.33997181e-01   7.04039478e+00   2.54547882e+01\n    1.70323145e+00]\n [  1.84289977e-01   3.15501571e+00   1.37455994e-04   3.64277154e-01\n    4.94003630e+00]]\n<NDArray 4x5 @cpu(0)>\n"
 }
]
```

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/744)
