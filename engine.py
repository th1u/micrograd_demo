import math

class Value:
    # 需要记录一些计算图
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0
        self._backward = lambda: None
    # 包装函数
    def __repr__(self):
        return f'Value(data={self.data})'
    # 每次运算过后 记录计算图中的节点
    # 加法
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0* output.grad
            other.grad += 1.0* output.grad
        output._backward = _backward
        return output
    # 乘法
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        output._backward = _backward
        return output
    # 检查交换后是否可以进行相乘
    def __rmul__(self, other):
        return self * other
    # 取反
    def __neg__(self):
        return self * (-1)
    # 减法
    def __sub__(self, other):
        return self + (-other)
    # 幂数
    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'only support int/float'
        output = Value(self.data **other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * output.grad
        output._backward = _backward
        return output
    # 除法
    def __truediv__(self, other):
        return self * other**-1
    # 指数
    def exp(self):
        x = self.data
        output = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += output.data * output.grad
        output._backward = _backward
        return output 
    # tanh
    def tanh(self):
        x = self.data
        t = (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
        output = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2)*output.grad
        output._backward = _backward
        return output
    # relu
    def relu(self):
        output = Value(0 if self.data < 0 else self.data, (self, ), 'ReLU')
        def _backward():
            self.grad += (output.data > 0) * output.grad
        output._backward = _backward
        return output
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    def __radd__(self, other):
        return self+other
    def __rsub__(self, other):
        return other + (-self)
    def __rtruediv__(self, other):
        return other * self**-1
    