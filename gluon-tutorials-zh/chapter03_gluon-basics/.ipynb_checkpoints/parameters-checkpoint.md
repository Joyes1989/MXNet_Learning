# 初始化模型参数

我们仍然用MLP这个例子来详细解释如何初始化模型参数。

```{.python .input  n=41}
from mxnet.gluon import nn
from mxnet import nd

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        # 两层的网络：一层输出为4的网络 + 一层输出为2的网络
        net.add(nn.Dense(4, activation="relu"))
        net.add(nn.Dense(2))
    return net

x = nd.random.uniform(shape=(3,5))
```

我们知道如果不`initialize()`直接跑forward，那么系统会抱怨说参数没有初始化。

```{.python .input  n=42}
import sys
try:
    net = get_net()
    print(net)
    # 会失败：net未进行权重Weight的初始化
    net(x)
except RuntimeError as err:
    sys.stderr.write(str(err))
```

```{.json .output n=42}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential(\n  (0): Dense(4, Activation(relu))\n  (1): Dense(2, linear)\n)\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Parameter sequential8_dense0_weight has not been initialized. Note that you should initialize parameters and create Trainer with Block.collect_params() instead of Block.params because the later does not include Parameters of nested child Blocks"
 }
]
```

正确的打开方式是这样

```{.python .input  n=43}
net.initialize()
print(net)
net(x)
```

```{.json .output n=43}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential(\n  (0): Dense(4, Activation(relu))\n  (1): Dense(2, linear)\n)\n"
 },
 {
  "data": {
   "text/plain": "\n[[-0.00289688  0.00205199]\n [ 0.0010183  -0.0016662 ]\n [-0.00048244  0.00038137]]\n<NDArray 3x2 @cpu(0)>"
  },
  "execution_count": 43,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 访问模型参数

之前我们提到过可以通过`weight`和`bias`访问`Dense`的参数，他们是`Parameter`这个类：
### 通过下标来访问多层网络的不同层，层下标从0开始：
    1. 这里的net[0]可以访问到网络的第一层（输出为0）
#### 对于每层的网络，可以通过weight、bias成员获取网络的参数：
    1. net[0]是一个dense网络，具有weight/bias参数
    2. 通过weight.data()、bias.data()来获取具体参数值
    3. 通过weight.grad()、bias.grad()来获取参数对应的梯度

```{.python .input  n=44}
w = net[0].weight
b = net[0].bias
print('name: ', net[0].name, '\nweight: ', w, '\nbias: ', b)
```

```{.json .output n=44}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "name:  sequential8_dense0 \nweight:  Parameter sequential8_dense0_weight (shape=(4, 5), dtype=<class 'numpy.float32'>) \nbias:  Parameter sequential8_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n"
 }
]
```

然后我们可以通过`data`来访问参数，`grad`来访问对应的梯度

```{.python .input  n=45}
print('weight:', w.data())
print('weight gradient', w.grad())
print('bias:', b.data())
print('bias gradient', b.grad())
```

```{.json .output n=45}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "weight: \n[[-0.06529249  0.02144811  0.06565464  0.02129445 -0.02506039]\n [-0.00960142 -0.03902322  0.05551652 -0.05022305 -0.01854134]\n [-0.05638361 -0.00897891  0.06776591  0.05486927 -0.03355227]\n [ 0.04286716  0.00518315  0.0285444  -0.00729033 -0.05596824]]\n<NDArray 4x5 @cpu(0)>\nweight gradient \n[[ 0.  0.  0.  0.  0.]\n [ 0.  0.  0.  0.  0.]\n [ 0.  0.  0.  0.  0.]\n [ 0.  0.  0.  0.  0.]]\n<NDArray 4x5 @cpu(0)>\nbias: \n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\nbias gradient \n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\n"
 }
]
```

### 我们也可以通过`collect_params`来访问Block里面所有的参数
    0. collect_params返回结果为dict类型，key为各变量名（格式为block.md中介绍的名字）
    1. （这个会包括所有的子Block）。
    2. 它会返回一个名字到对应Parameter的dict。
    3. 既可以用正常`[]`来访问参数，也可以用`get()`，它不需要填写名字的前缀。

```{.python .input  n=47}
params = net.collect_params()
print(params)
# sequential8_dense0_bias 这个名字是系统默认生成的，可参考block.md
print(params['sequential8_dense0_bias'].data())
print(params.get('dense0_weight').data())
```

```{.json .output n=47}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "sequential8_ (\n  Parameter sequential8_dense0_weight (shape=(4, 5), dtype=<class 'numpy.float32'>)\n  Parameter sequential8_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential8_dense1_weight (shape=(2, 4), dtype=<class 'numpy.float32'>)\n  Parameter sequential8_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)\n\n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\n\n[[-0.06529249  0.02144811  0.06565464  0.02129445 -0.02506039]\n [-0.00960142 -0.03902322  0.05551652 -0.05022305 -0.01854134]\n [-0.05638361 -0.00897891  0.06776591  0.05486927 -0.03355227]\n [ 0.04286716  0.00518315  0.0285444  -0.00729033 -0.05596824]]\n<NDArray 4x5 @cpu(0)>\n"
 }
]
```

## 使用不同的初始函数来初始化

    1. 我们一直在使用默认的`initialize`来初始化权重（除了指定GPU `ctx`外）。它会把所有权重初始化成在`[-0.07, 0.07]`之间均匀分布的随机数。我们可以使用别的初始化方法。例如使用均值为0，方差为0.02的正态分布
    2. 通过制定init参数的值，来设置不同的初始化方式

```{.python .input  n=52}
from mxnet import init
# sigma -- σ（西格玛）指标准差
# 重复初始化需要制定force_reinit=True
params.initialize(init=init.Normal(sigma=0.02), force_reinit=True)
print(net[0].weight.data(), net[0].bias.data())
```

```{.json .output n=52}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[-0.00174615 -0.02994123 -0.01775906  0.01515305  0.01055657]\n [-0.00683027 -0.01343734 -0.02067723  0.01186648  0.00047213]\n [-0.00053659 -0.00901016  0.01337759 -0.01849548 -0.02293071]\n [-0.00498697  0.02817637 -0.01538535  0.04250335 -0.01058471]]\n<NDArray 4x5 @cpu(0)> \n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\n"
 }
]
```

看得更加清楚点：

```{.python .input  n=53}
params.initialize(init=init.One(), force_reinit=True)
print(net[0].weight.data(), net[0].bias.data())
```

```{.json .output n=53}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 1.  1.  1.  1.  1.]\n [ 1.  1.  1.  1.  1.]\n [ 1.  1.  1.  1.  1.]\n [ 1.  1.  1.  1.  1.]]\n<NDArray 4x5 @cpu(0)> \n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\n"
 }
]
```

更多的方法参见[init的API](https://mxnet.incubator.apache.org/api/python/optimization.html#the-mxnet-initializer-package). 下面我们自定义一个初始化方法。

```{.python .input}
class MyInit(init.Initializer):
    def __init__(self):
        super(MyInit, self).__init__()
        # 控制输出细节信息
        self._verbose = True
    def _init_weight(self, _, arr):
        # 初始化权重，使用out=arr后我们不需指定形状
        print('init weight', arr.shape)
        nd.random.uniform(low=5, high=10, out=arr)
    def _init_bias(self, _, arr):
        print('init bias', arr.shape)
        # 初始化偏移
        arr[:] = 2

# FIXME: init_bias doesn't work
params.initialize(init=MyInit(), force_reinit=True)
print(net[0].weight.data(), net[0].bias.data())
```

## 延后的初始化

我们之前提到过Gluon的一个便利的地方是模型定义的时候不需要指定输入的大小，在之后做forward的时候会自动推测参数的大小。我们具体来看这是怎么工作的。

新创建一个网络，然后打印参数。你会发现两个全连接层的权重的形状里都有0。 这是因为在不知道输入数据的情况下，我们无法判断它们的形状。

```{.python .input}
net = get_net()
print(net.collect_params())
```

然后我们初始化

```{.python .input}
net.initialize(init=MyInit())
```

你会看到我们并没有看到MyInit打印的东西，这是因为我们仍然不知道形状。真正的初始化发生在我们看到数据时。

```{.python .input}
net(x)
```

这时候我们看到shape里面的0被填上正确的值了。

```{.python .input}
print(net.collect_params())
```

## 避免延后初始化

有时候我们不想要延后初始化，这时候可以在创建网络的时候指定输入大小。

```{.python .input}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, in_units=5, activation="relu"))
    net.add(nn.Dense(2, in_units=4))

net.initialize(MyInit())
```

## 共享模型参数

有时候我们想在层之间共享同一份参数，我们可以通过Block的`params`输出参数来手动指定参数，而不是让系统自动生成。

```{.python .input}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, in_units=4, activation="relu"))
    net.add(nn.Dense(4, in_units=4, activation="relu", params=net[-1].params))
    net.add(nn.Dense(2, in_units=4))


```

初始化然后打印

```{.python .input}
net.initialize(MyInit())
print(net[0].weight.data())
print(net[1].weight.data())
```

## 总结

我们可以很灵活地访问和修改模型参数。

## 练习

1. 研究下`net.collect_params()`返回的是什么？`net.params`呢？
1. 如何对每个层使用不同的初始化函数
1. 如果两个层共用一个参数，那么求梯度的时候会发生什么？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/987)
