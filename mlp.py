"""
Script for building theano graph for MLP
Siddharth Sigia
Feb,2014
C4DM
"""
import sys
import os
import numpy
import theano
import theano.tensor as T 
from theano.tensor.shared_randomstreams import RandomStreams
import pdb

class MLP():
	def __init__(self,n_inputs=513,n_outputs=10,n_hidden=[50,50,50],activation='sigmoid',output_layer='sigmoid',dropout_rates=None):
		self.x = T.matrix('x')
		self.y = T.matrix('y')
		self.n_layers = len(n_hidden)
		self.n_inputs = n_inputs
		self.n_hidden = n_hidden
		self.n_outputs = n_outputs
		self.sizes = [self.n_inputs] + self.n_hidden + [self.n_outputs]
		self.numpy_rng = numpy.random.RandomState(123)
		self.theano_rng = RandomStreams(self.numpy_rng.randint(2**10))
		self.output_layer = output_layer
		if dropout_rates is not None:
			assert len(dropout_rates)==len(n_hidden), "dropout_rates masks must be specified for each of the layers."
			self.dropout_rates = dropout_rates
		self.initialize_params()
		self.set_activation(activation)
		if not self.dropout_rates:
			self.build_graph()
		else:
			self.build_graph_dropout()

	def initialize_params(self,):
		self.W = []
		self.b = []
		for i in xrange(len(self.sizes)-1):
			input_size = self.sizes[i]
			output_size = self.sizes[i+1]
			W_init = numpy.asarray(self.numpy_rng.uniform(low=-4*numpy.sqrt(6./(input_size+output_size)),
				                                          high=4*numpy.sqrt(6./(input_size+output_size)),
				                                          size=(input_size,output_size)),
														  dtype=theano.config.floatX)
			W = theano.shared(value=W_init,name='W_%d'%(i),borrow=True)
			b = theano.shared(value=numpy.zeros(output_size,dtype=theano.config.floatX),name='b_%d'%(i),
							  borrow=True)
			self.W.append(W)
			self.b.append(b)
		self.params = self.W + self.b

	def set_activation(self,activation):
		if activation=='sigmoid':
			self.activation = lambda x:T.nnet.sigmoid(x)
		elif activation=='ReLU':
			self.activation = lambda x:T.maximum(0.,x)
		else:
			print 'Activation must be sigmoid or ReLU. Quitting'
			sys.exit()

	def fprop(self,inputs,output_layer='sigmoid'):
	 	h = []
	 	h.append(self.activation(T.dot(inputs, self.W[0]) + self.b[0]))
	 	for i in xrange(1,len(self.n_hidden)):
	 		h.append(self.activation(T.dot(h[-1], self.W[i]) + self.b[i]))
	 	if output_layer=='sigmoid':
	 		h.append(T.nnet.sigmoid(T.dot(h[-1], self.W[-1]) + self.b[-1]))
	 	elif output_layer=='softmax':
	 		h.append(T.nnet.softmax(T.dot(h[-1], self.W[-1]) + self.b[-1]))
	 	else:
	 		print 'Output layer must be either sigmoid or softmax. Quitting.'
	 		sys.exit()
	 	return h[-1]


	def fprop_dropout(self,inputs,output_layer='sigmoid'):
		h = []
		r = 1./(1-self.dropout_rates[0])
		mask = T.cast(self.theano_rng.binomial(size=inputs.shape,n=1,p=1-self.dropout_rates[0]),theano.config.floatX) 
		corr_input = inputs * mask
		h.append(self.activation(r * T.dot(corr_input,self.W[0]) + self.b[0]))
		for i in xrange(1,len(self.n_hidden)):
	 		r = 1./(1-self.dropout_rates[i])
	 		mask = T.cast(self.theano_rng.binomial(size=h[i-1].shape,n=1,p=1-self.dropout_rates[i]),theano.config.floatX) 
	 		corr_input = h[i-1] * mask
	 		h.append(self.activation(r * T.dot(corr_input,self.W[i]) + self.b[i]))
	 	if output_layer=='sigmoid':
	 		h.append(T.nnet.sigmoid(T.dot(h[-1], self.W[-1]) + self.b[-1]))
	 	elif output_layer=='softmax':
	 		h.append(T.nnet.softmax(T.dot(h[-1], self.W[-1]) + self.b[-1]))
	 	else:
	 		print 'Output layer must be either sigmoid or softmax. Quitting.'
	 		sys.exit()
	 	return h[-1]

	def build_graph(self,):
		self.z = self.fprop(self.x,output_layer='sigmoid')
		#L = -T.sum(self.z*T.log(self.y) + (1-self.z)*T.log(1-self.y),axis=1)
		L = -T.sum(self.y*T.log(self.z) + (1-self.y)*T.log(1-self.z),axis=1)
		#L = T.sum((self.z-self.y)**2,axis=1)
		self.cost = T.mean(L)
		self.preds = T.max(self.z,axis=1)
		self.acc = T.neq(T.argmax(self.z,axis=1),T.argmax(self.y,axis=1)).mean()
		#return [self.x,self.y],self.cost,self.params

	def build_graph_dropout(self,):
		self.z_dropout = self.fprop_dropout(self.x,output_layer='sigmoid')
		self.z = self.fprop(self.x,output_layer='sigmoid')
		L = -T.sum(self.y*T.log(self.z_dropout) + (1-self.y)*T.log(1-self.z_dropout),axis=1)
		self.cost = T.mean(L)
		self.acc = T.neq(T.argmax(self.z,axis=1),T.argmax(self.y,axis=1)).mean()

if __name__=='__main__':
	test = MLP()
	_,_,_ = test.build_graph()
	