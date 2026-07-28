from Autograd import Value , draw_dot
import torch
import random

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

class Layer:
	def __init__(self , n_in , n_out):
		self.neurons = [Neuron(n_in) for _ in range(n_out)]
		
	def __call__(self , x):
		return [neuron(x) for neuron in self.neurons]
	
	def parameters(self):
	    return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
	def __init__(self , n_in , n_outs):
		sz = [n_in] + n_outs
		self.layers = [Layer(sz[i] , sz[i+1]) for i in range(len(sz))]
		
	def __call__(self , x):
		for layer in self.layers:
			x = layer(x)
		self.out = x
		return self.out
		
	def parameters(self):
		return [p for layer in self.layers for p in layer.parameters]

class Neuron_torch:
	def __init__(self , n_in):
		self.weights = torch.randn((n_in,) , requires_grad = True) # torch.randn(size) initializes a tensor.
		# by default , variables defined by pytorch has requires_grad = False ie. pytorch doesn't keep a track of it's manipulations , thus can't find it's gradient , we need to explicitely mention it to keep track of it.
		self.bias = torch.zeros((1,) , requires_grad = True)
	def __call__(self , x):
		self.out = x @ self.weights + self.bias
		return self.out
	def parameters(self):
		return self.weights + [self.bias]



