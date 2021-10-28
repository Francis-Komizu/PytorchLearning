# Tensors

## What is Tensors?

>Tensors are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.
>
>Tensors are similar to NumPy’s ndarrays, except that tensors can run on GPUs or other specialized hardware to accelerate computing. 

张量（tensor) 可以和之前学的矢量和矩阵相联系。

- 0阶张量是一个数
- 1阶张量是一个矢量(vector)，计算机中就可以用一位数组表示
- 2阶张量是一个矩阵
- ……

下面是一个3阶张量：

[
	[[9,1,8],[6,7,5],[3,4,2]],
	[[2,9,1],[8,6,7],[5,3,4]],
	[[1,5,9],[7,2,6],[4,8,3]]
]

中间的每一行都是一个矩阵（2阶张量）

另外，3阶张量又叫”空间矩阵“或者”三维矩阵“，例如：

![img](https://pic1.zhimg.com/80/v2-94f7732383f7b46a0c2ec5108e1fe088_720w.jpg)




---

## Tensor Initialization

**Directly from data**

>Tensors can be created directly from data. The data type is automatically inferred.

```python
data = [[1,2,3],[4,5,6],[7,8,9]]
x_data = torch.tensor(data)
```

OUt:

```python
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
```



**From a NumPy array**

> Tensors can be created from NumPy arrays.

```python
data = [[1,2,3],[4,5,6],[7,8,9]]
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)
```

Out:

```python
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]], dtype=torch.int32)
```



And vica versa, NumPy arrays can also be created from tensors.

```python
data = [[1,2,3],[4,5,6],[7,8,9]]
x_t = torch.tensor(data)
np_array = x_t.numpy()
print(np_array)
```

Out:

```python
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```



> Tensors on the CPU and NumPy arrays can share their underlying memory locations, and ==changing one will change the other==.

```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

Out:

```python
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
```



A change in the tensor reflects in the NumPy array.

```python
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

Out:

```python
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
```



**From another tensor**

除非显式更改，新创建的张量会保持原张量的性质（性状、数据类型）。

> The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.

```python
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
```

Out：

```python
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.1148, 0.1487],
        [0.0268, 0.5634]])
```



**With random or constant values**

shape决定张量的维度。

> shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.

```python
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

Out:

```python
Random Tensor:
 tensor([[0.3085, 0.9156, 0.5904],
        [0.0319, 0.0674, 0.5835]])

Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```



---

## Tensor Attributes

> Tensor attributes describe their shape, datatype, and ==the device on which they are stored==.

```python
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

Out:

```python
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```



---

## Tensor Operations

**Move our tensor to the GPU**

```python
# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")
```

Out:

```python
pyDevice tensor is stored on: cuda:0
```



**Standard numpy-like indexing and slicing:**

```python
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)
```

Out:

```python
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```



**Joining tensors** 

> You can use `torch.cat` to concatenate a sequence of tensors along a given dimension.

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

Out:

```python
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```



**Multiplying tensors**

张量中对应元素相乘。

```python
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")
```

Out:

```python
tensor.mul(tensor)
 tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor * tensor
 tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```



矩阵乘法

```python
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")
```

Out:

```python
tensor.matmul(tensor.T)
 tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])

tensor @ tensor.T
 tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
```



**In-place operations**

> Operations that have a `_` suffix are in-place. For example: `x.copy_(y)`, `x.t_()`, will change `x`.

in-place指的是”就地“，也就是操作过后会改变x。

```python
print(tensor, "\n")
tensor.add_(5)
print(tensor)
```

Out:

```python
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
```
