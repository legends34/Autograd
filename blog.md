Inspired from Andrej Karpathy's [# The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) This blog covers building a Neural network and Autograd from scratch.
## Why Autograd?
Every Layer of a  Neural Network at it's core is a composite functions -> f(g(...(x))) , and Neural Networks can go deeper and deeper , with a lot of parameters . We can't always sit and derive the gradient equation for each of the parameter , more-over , we need a method that can be generalized to any Neural Network , thus Autograd - a way to automatically calculate the gradient of functions.

## The Aproach
What comes to your mind when you think about differentiating a composite function like f(g(x))? Chain rule?
#### The Chain Rule
$$\frac{d(F)}{d(x)} = \frac{d(F)}{d(G)} \cdot \frac{d(G)}{d(x)} \quad $$
but for this , we need to store the previous values , and there derivatives , thus will have to define a custom data structure
```python
class Value:
	def __init__(self , data , _prev = set() , _op = '' , label = ''):
		self.data = data
		self._prev = _prev
		self._op = _op
		self.grad = 0
		self._backward = None
		self.label = label
```
prev : is a set to ensure we don't get repeated elements ( y = x * x) , we don't want to store (x ,  x) with different derivatives.
op : stores the operation performed by the prev to get to the output
grad : stores the gradient
backward : stores the function that shall be used to find the gradient
label : for reference

### Operations
Now since we have a different datatype , we will have to define the various mathematical operations. And along with the operations , we will have to define the gradients.

```python
def __add__(self , other):
	other = other if isinstance(other , Value) else Value(other) # Ensure it is of the same dataype
	out.data = self.data + other.data
	out._pre = set(self , other)
	
	def _backward(): # Function to calculate the gradient
		self.grad += 1 * out.grad 
		other.grad += 1 * out.grad
	out._backward = _backward   # We are not calling the function , just saving it , can be called later on
	
```

$$ self.grad = \frac{d(out)}{d(self)} $$
out._ backward will propagate the gradient to self


## Backprop
in order to back propagate , we will need every node in reversed order . We can use topological sort to do this

```python
topo = []
visited = set()

def build_topo(v):
	if v not in visited:
		visited.add(v)
		for child in v._prev:
			build_topo(child)
		topo.append(v)
```

We will get the nodes sorted as parents -> child
** topo.append(v) should not be placed before the loop , other wise we won't get the sorted 
```
	A (Input)
   / \
  B   C
   \ /
    D (Output)
```

we will get output DBAC instead of DBCA as we will be appending the parent first , and then going to look for child , without exploring all the nodes at the current level.

Finally , for backpropagation , we can add backwards function to the Value class.
```python
def backwards(self):
	self.grad = 1
	''' Topological sort '''
	build_topo(self)
	
	for node in reversed(topo):
		node._backward()  # function we had stored in the Value class
```

Therefore , now , we can write any composite function , and at last call out.backwards() , and we will get the differentiation of all the functions with respect to all the functions.


### Visualization
we can use Digraph module from graphviz library to view our manipulations as nodes and edges.
	`This is completely for visualization , and we are directly using Library functions , for details , you can read it's documentation.`

```python
from graphviz import Digraph

def trace(root): # we will use this function to map out all the nodes and edges
	nodes , edges = set() , set()
	def build(v):
		if v not in nodes:
			nodes.add(root)
			for child in root._prev:
				edges.add((v , child))
				trace(child)
	build(root)
	
	return nodes , edges
	
def draw_dot(root):
	dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # left to right
	nodes , edges = trace(root)
	
	for n in nodes:
		uid = str(id(n))
		
		dot.node(name = uid , label = "%s | data = %s | grad = %s" %{n.label , n.data , n.grad} , shape = 'round')
		if n._op:
			dot.node(name = uid + n._op , n.label)
			dot.edge(uid + n.op , uid)
		for n1 , n2 in edges:   # tuple unpacking , we don't need nested loop
			dot.edge(str(id(n1)) , str(id(n2)) + n2._op)
		
		return dot
```


### Building a Neural Network

Now that we have implemented Autograd , we can go ahead to implement a Neural Network from scratch.
## A Single Neuron
The first step to building a Neural Network is obviously - The Neuron. So , what defines a Neuron? There are a bunch on weights and biases , together called parameters. the input values we enter are linearly transformed using these parameters to output a single value. The parameters are first defined randomly , then trained by updating the weights.

The forward Pass:
$$ \hat{y}= \sum{(w_i \cdot x_i)} + b$$
where $\hat{y}$ is the predicted value and $wi's$ are the weights , $xi's$ are the input values , and $b$ is the bias .

```python
class Neuron:
	def __init__(self , n_in):
		self.weights = [Value(random.uniform(-1 ,1)) for _ in range(n_in)] # Note that the no. of weights equals the no. inputs
		self.bias = Value(random.uniform(-1 ,1))
		# Randomly initialized the parameters using the random library and typecasted them to Value class
	def __call__(self , x):  # The forward pass
		self.out = sum((wi*xi for wi , xi in zip(self.weights , x)) , self.bias)
		return self.out
	def parameters(self):  # This will be used to return the parameters
		return self.weights + [self.bias]
```
We initialize the parameters in the range (-1 , 1) due to two reasons : 
1) To get both positive and negative nos. , so that neurons fire differently in the starting epoches of training
2) To scale the weights - if they are initialized large , they may cause vanishing / Exploding gradients int the activation functions , small weights keep them in the sweet spot.

### Layer
Once we have a Neuron , We need to define a Layer of Neurons. A Layer is basically a bunch of neurons stacked together , they take the same inputs , but output different values depending on their parameters.
Here we are taking two inputs - n_in : the no. of input each neuron will have (this is common to all the neurons) , n_out : "The no. of outputs we want" or in simple terms , the no. of neurons we want in the layer.
for initializing the layer , we will be calling the Neuron class we created above , and storing all the neurons in a list.
for the forward pass , we will be iterating through each neuron , pass the input x and again store the outputs in a list.

```python
class Layer:
	def __init__(self , n_in , n_out , torch = False):
		self.neurons = [Neuron(n_in) for _ in range(n_out)] if not torch else [Neuron_torch(n_in) for _ in range(n_out)]
		
	def __call__(self , x):
		return [neuron(x) for neuron in self.neurons]
	
	def parameters(self):
	return [p for neuron in self.neurons for p in neuron.parameters()]
```

### MLP
Now we need to Sequentially Arrange the layers so that the ${(n+1)}^{th}$ layer's input is the $n^{th}$ layer's output , to Finally get our very own Neural Network (Multi Layer Perceptron or MLP), built from scratch.
So , the inputs we will be requiring here are : 
1) n_in : no. of input to the first layer
2) n_1 : no. of neurons in the 1st layer
3) n_2..
but , this way , we won't have a defined no. of inputs. We can do this , by passing the n_out of each layer as a list "n_outs"
The definition of each layer has to be done such that the n_into the ${(n+1)}^{th}$ layer is the n_out
of the $n^{th}$ layer.

For initializing the MLP , we will be calling the Layer class defined earlier , and store them in a list. we will also create another list , which we can iterate to assign the no. of inputs and outputs to each layer accordingly.

For the forward pass , the input of the ${(n+1)}^{th}$ layer has to be the output of the $n^{th}$ layer. for this , we will be iterating through all the layers , and assign x to be equal to the output of the layer , this way , each time we are passing the output of the previous layer to the current layer , and then storing them as input for the next layer (check the code).

```python
class MLP:
	def __init__(self , n_in , n_outs):
		sz = [n_in] + n_outs
		self.layers = [Layer(sz[i] , sz[i+1]) for i in range(len(sz))]
		
	def __call__(slef , x):
		for layer in self.layers:
			x = layer(x)
		self.out = x
		return self.out
		
	def parameters(self):
		return [p for layer in self.layers for p in layer.parameters]
```

And Finally , We have our own Neural Network - well.. No. 
We have defined our Neural Network , but the parameters are random , and thus it will output completely random values. We need to train our Network in order for it to output meaningful results.

### Training loop
Training a Neural Network is quite similar to how you train a simple linear / logistic regression model. You start with a labeled dataset , pass the inputs to the model , and get it's prediction.
Then define a Loss function to know quantitively know how wrong our model is , then update the weights and repeat.
```python
X , Y =  # Data , X is the input and Y is the output
n_in = 
n_outs = []
model = MLP(n_in , n_outs) # defining The model
epochs = # no. of epoches to train
lr = # Learning rate

for epoch in range(epochs):
	# forward pass
	y_hat = model(X) # making predictions.
	loss = loss_function() # define loss according to the problem
	# making gradients 0 before calculating (otherwise , the gradients will keep on getting added at each epoch (self.grad += gradient)*)
	for p in model.parameters():
		p.grad = 0
	
	# backward pass
	loss.backwards()
	
	# update
	for p in model.parameters():
		p.data -= lr * p.grad
```

And now , finally , we can put our own Neural Network to use. But there's one thing to note - our Value class takes scalers as input , but for almost all practical purposes , we will be working with tensors. Here we are sequentially looping through all the features , which is highly inefficient. Tensors are made for parallel computation. So it is always better to use libraries like pytorch or tensorflow while working with neural networks.
Below is the pytorch implementation of the Neural Network , the working remains same , just that we are replacing our Value class with torch's  tensor implementation, I have tried to explain each library function , but if you still have doubts , you can visit the pytorch documentation.

we will be importing torch
```python
import torch
```
### Neuron
```python
class Neuron:
	def __init__(self , n_in):
		self.weights = torch.randn((n_in,) , requires_grad = True) # torch.randn(size) initializes a tensor.
		# by default , variables defined by pytorch has requires_grad = False ie. pytorch doesn't keep a track of it's manipulations , thus can't find it's gradient , we need to explicitely mention it to keep track of it.
		self.bias = torch.zeros((1,) , requires_grad = True)
	def __call__(self , x):
		self.out = x @ self.weights + self.bias
		return self.out
	def parameters(self):
		return self.weights + [self.bias]
```

The Layer and MLP classes will remain the same.

### Training loop
```python
# while defining the model , ensure torch = True
model = MLP(n_in , n_outs , torch = True)
for epoch in range(epochs):
	# forward pass
	y_hat = model(X) # making predictions.
	loss = loss_function() # define loss according to the problem
	# pytorch initialies gradients = None , they need to be zeroes , after that , pytorch handles the gradient (no nee to make them zero at every epoch)
	for p in model.parameters():
		if p.grad is not None:
			p.grad.zero_()
	
	# backward pass
	loss.backward()
	
	# update
	with torch.no_grad():
		for p in model.parameters():
			p.data -= lr * p.grad
```

Or you can directly use torch.nn.Linear() for a standard Neural Network.
