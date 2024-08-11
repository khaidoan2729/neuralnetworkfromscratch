import random
import math
import torch


class Value: 
    def __init__(self, data, _children=(), _op='', label=''):   #label for graphical view, not really important
        self.data = data                #Value data 
        self.grad = 0.00                #Rate of change of result 
        self._backward = lambda: None   #Initially, _backward is function that does nothing, such as leaf node
        self._prev = set(_children)     #Set for no duplicate
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"   #For displaying (JupyterLab/Notebook)
    
    #Addition
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')     #Current node and sibling node becomes children of resulting node
        
        def _backward(): 
            self.grad += 1.0 * out.grad         #d(self+other)/d(self) = 1.0 ; out.grad is Chain rule
            other.grad += 1.0 * out.grad        #Gradient is accumulative                        
        out._backward = _backward

        return out
    
    #Do a + 3 (valid) when given 3 + a (invalid)
    def __radd__(self, other): 
        return self + other

    def __neg__ (self): 
        return self*(-1)

    def __sub__(self, other): 
        return self + (-other)
    
    #Multiplication
    def __mul__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')     #Current node and sibling node becomes children of resulting node
        
        def _backward():
            self.grad += out.grad * other.data  #d(self*other)/d(self) = other ; out.grad is Chain rule
            other.grad += out.grad * self.data
        out._backward = _backward

        return out

    #Do a * 3 (valid) when given 3 * a (invalid)
    def __rmul__(self, other): 
        return self * other 

    #Tanh(x) for activatio function
    def tanh(self):     # tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        out_data = (math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
        out = Value(out_data, (self, ), 'tanh')

        def _backward():    
            self.grad += (1 - out.data**2) * out.grad    #d(tanh(x))/dx = 1 - (tanh(x))^2 ; out.grad is Chain rule 
        out._backward = _backward

        return out
    
    #Division
    def __truediv__ (self, other): 
        return self * (other**-1)

    #Power
    def __pow__ (self, other): 
        assert isinstance(other, (int, float)), "only support int/float powers"
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad  #d(a^b)/da = b* (a^(b-1)) ; out.grad is Chain rule
        out._backward = _backward

        return out

    #Exponential (natural log)
    def exp(self):
        out_data = math.exp(self.data)
        out = Value(out_data, (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad    
        out._backward = _backward

        return out

    #Backpropagation
    def backward(self):         
        topo = []           
        visited = set()                     #Append recursively child nodes to list
        def build_topo(v): 
            if v not in visited: 
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)    

        self.grad = 1.0                     #Set grad of current node to 1.0 (before performing backpropagation)

        for node in reversed(topo):         #Backpropagation
            node._backward()



class Neuron: 
    def __init__(self, num_in): # constructor takes number of inputs 
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(num_in)]      #Create random weights [-1,1] for inputs
        self.bias = Value(random.uniform(-1,1))                                #Create random biases [-1,1] for inputs (trigger happiness)

    def __call__ (self, x): #For passing arguments x
        '''
        Example: 
        x = [1.0, 2.0]
        neu = Neuron(2)
        neu(x)
        '''
        w_input_tuple = zip(self.weights, x)
        act = Value(0.0)
        for w_i, x_i in w_input_tuple: 
            act = act + w_i*x_i         #Dot product of vector weights and input 

        act = act + self.bias           #Add bias 
        out = act.tanh()                #tanh(x) for activation

        return out

    def parameters (self): 
        return self.weights + [self.bias]   #return weights and bias as params


class Layer:

    def __init__ (self, num_in, num_out):
        self.neurons = [Neuron(num_in) for _ in range(num_out)]

    def __call__ (self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs)==1 else outs
    
    def parameters (self): 
        params = []
        for neuron in self.neurons: 
            params.extend(neuron.parameters())
        return params

class MLP:      #Multi-layered perceptron: Layers feed into each other sequentially
    
    def __init__ (self, num_in, num_outs):  #take number of inputs and LIST of size of layers in the MLP
        sizes = [num_in] + num_outs
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(num_outs))]

    def __call__ (self, x):
        for layer in self.layers: 
            x = layer(x)
        return x

    def parameters (self):
        return [p for layer in self.layers for p in layer.parameters()]
            


'''
#x = [2.0, 3.0, -1.0]
n = MLP(3, [4,4,1])
xs = [
    [2.0, 3.0, -1.0], 
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]       #inputs

ys = [1.0, -1.0, -1.0, 1.0]     #targets outputs
learning_rate = -1

for i in range(200):
    # Forward pass 
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    #Zeno Gradient
    for p in n.parameters():
        p.grad = 0.0        #Reset gradient before backward pass

    # Backward pass
    loss.backward()

    learning_rate = 0.9*learning_rate

    # Update
    for p in n.parameters():
        p.data += learning_rate * p.grad

    print(i, loss.data)

print(ypred)

x = torch.tensor([1,2,3,4])
print(x)'''
