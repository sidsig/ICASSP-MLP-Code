"""
Trainer class
Siddharth Sigia
Feb,2014
C4DM
"""
import os
import sys
import numpy
import theano
import pickle
from preprocessing import PreProcessor
from mlp import MLP
from sgd import SGD_Optimizer
from dataset import Dataset
import state
import pdb

class trainer():
	def __init__(self,state):
		self.state = state
		self.dataset_dir = self.state.get('dataset_dir','')
		self.list_dir = os.path.join(self.dataset_dir,'lists')
		self.lists = {}
		self.lists['train'] = os.path.join(self.list_dir,'train_1_of_1.txt')
		self.lists['valid'] = os.path.join(self.list_dir,'valid_1_of_1.txt')
		self.lists['test'] = os.path.join(self.list_dir,'test_1_of_1.txt')
		self.preprocessor = PreProcessor(self.dataset_dir) 
		print 'Preparing train/valid/test splits'
		self.preprocessor.prepare_fold(self.lists['train'],self.lists['valid'],self.lists['test'])
		self.data = self.preprocessor.data
		self.targets = self.preprocessor.targets
		print 'Building model.'
		self.model = MLP(n_inputs=self.state.get('n_inputs',513),n_outputs=self.state.get('n_ouputs',10),
						 n_hidden=self.state.get('n_hidden',[50]),activation=self.state.get('activation','sigmoid'),
						 output_layer=self.state.get('sigmoid','sigmoid'),dropout_rates=self.state.get('dropout_rates',None))

	def train(self,):
		print 'Starting training.'
		print 'Initializing train dataset.'
		self.batch_size = self.state.get('batch_size',20)
		train_set = Dataset([self.data['train']],batch_size=self.batch_size,targets=[self.targets['train']])
		print 'Initializing valid dataset.'
		valid_set = Dataset([self.data['valid']],batch_size=self.batch_size,targets=[self.targets['valid']])
		self.optimizer = SGD_Optimizer(self.model.params,[self.model.x,self.model.y],[self.model.cost,self.model.acc],momentum=self.state.get('momentum',False))
		lr = self.state.get('learning_rate',0.1)
		num_epochs = self.state.get('num_epochs',200)
		save = self.state.get('save',False)
		mom_rate = self.state.get('mom_rate',None)
		self.optimizer.train(train_set,valid_set,learning_rate=lr,num_epochs=num_epochs,save=save,mom_rate=mom_rate)

if __name__=='__main__':
	state = state.get_state()
	test = trainer(state)
	test.train()