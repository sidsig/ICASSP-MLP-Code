"""
SGD optimizer class 
Siddharth Sigtia
Feb,2014
C4DM
"""
import numpy, sys
import theano
import theano.tensor as T
import cPickle
import os
from theano.compat.python2x import OrderedDict
import copy
import pdb


class SGD_Optimizer():
    def __init__(self,params,inputs,costs,updates_old=None,consider_constant=[],momentum=True):
        """
        params: parameters of the model
        inputs: list of symbolic inputs to the graph
        costs: list of costs to be evaluated. The first element MUST be the objective.
        updates_old: OrderedDict from previous graphs that need to be accounted for by SGD, typically when scan is used.
        consider_constant: list of theano variables that are passed on to the grad method. Typically RBM.
        """
        self.inputs = inputs
        self.params = params
        self.momentum = momentum
        if self.momentum:
            self.params_mom = []
            for param in self.params:
                param_init = theano.shared(value=numpy.zeros(param.get_value().shape,dtype=theano.config.floatX),name=param.name+'_mom')
                self.params_mom.append(param_init)
        self.costs = costs 
        self.num_costs = len(costs)
        assert (isinstance(costs,list)), "The costs given to the SGD class must be a list, even for one element."
        self.updates_old = updates_old
        self.consider_constant = consider_constant
        self.build_train_fn()

    def build_train_fn(self,):
        self.lr_theano = T.scalar('lr')
        self.grad_inputs = self.inputs + [self.lr_theano]
        if self.momentum:
            self.mom_theano = T.scalar('mom')
            self.grad_inputs = self.grad_inputs + [self.mom_theano]
        
        self.gparams = T.grad(self.costs[0],self.params,consider_constant=self.consider_constant)
        if not self.momentum:
            print 'Building SGD optimization graph without momentum'
            updates = OrderedDict((i, i - self.lr_theano*j) for i, j in zip(self.params, self.gparams))
        else:
            print 'Building SGD optimization graph with momentum'
            updates = OrderedDict()
            for param,param_mom,gparam in zip(self.params,self.params_mom,self.gparams):
                param_inc = self.mom_theano * param_mom - self.lr_theano * gparam
                updates[param_mom] = param_inc
                updates[param] = param + param_inc
        self.calc_cost = theano.function(self.inputs,self.costs)
        if self.updates_old:
            updates_old = copy.copy(updates_old) #To avoid updating the model dict if updates dict belongs to model class, very unlikely case.
            self.updates_old.update(updates)
        else:
            self.updates_old = OrderedDict()
            self.updates_old.update(updates)

        self.f = theano.function(self.grad_inputs, self.costs, updates=self.updates_old)

    def train(self,train_set,valid_set=None,learning_rate=0.1,num_epochs=500,save=False,output_folder=None,lr_update=None,mom_rate=0.9):
        self.best_cost = numpy.inf
        self.init_lr = learning_rate
        self.lr = numpy.array(learning_rate)
        self.mom_rate = mom_rate
        self.output_folder = output_folder
        self.train_set = train_set
        self.valid_set = valid_set
        self.save = save
        self.lr_update = lr_update
        try:
            for u in xrange(num_epochs):
                cost = []
                for i in self.train_set.iterate(True): 
                    inputs = i + [self.lr]
                    if self.momentum:
                        inputs = inputs + [self.mom_rate]
                    cost.append(self.f(*inputs))
                mean_costs = numpy.mean(cost,axis=0)
                print '  Epoch %i   ' %(u+1)
                print '***Train Results***'
                for i in xrange(self.num_costs):
                    print "Cost %i: %f"%(i,mean_costs[i])

                if not valid_set:
                    this_cost = numpy.absolute(numpy.mean(cost, axis=0))
                    if this_cost < best_cost:
                        best_cost = this_cost
                        print 'Best Params!'
                        if save:
                            self.save_model()
                    sys.stdout.flush()     
                else:
                    self.perform_validation()
                
                if lr_update:
                    self.update_lr(u+1,begin_anneal=1)

        except KeyboardInterrupt: 
            print 'Training interrupted.'
    
    def perform_validation(self,):
        cost = []
        for i in self.valid_set.iterate(True): 
            cost.append(self.calc_cost(*i))
        mean_costs = numpy.mean(cost,axis=0)
        print '***Validation Results***'
        for i in xrange(self.num_costs):
            print "Cost %i: %f"%(i,mean_costs[i])
     
        this_cost = numpy.absolute(numpy.mean(cost, axis=0))[1] #Using accuracy as metric
        if this_cost < self.best_cost:
            self.best_cost = this_cost
            print 'Best Params!'
            if self.save:
                self.save_model()

    def save_model(self,):
        best_params = [param.get_value().copy() for param in self.params]
        if not self.output_folder:
            cPickle.dump(best_params,open('best_params.pickle','w'))
        else:
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
            save_path = os.path.join(self.output_folder,'best_params.pickle')
            cPickle.dump(best_params,open(save_path,'w'))


    def update_lr(self,count,update_type='annealed',begin_anneal=500.,min_lr=0.01,decay_factor=1.2):
        if update_type=='annealed':
            scale_factor = float(begin_anneal)/count
            self.lr = self.init_lr*min(1.,scale_factor)
        if update_type=='exponential':
            new_lr = float(self.init_lr)/(decay_factor**count)
            if new_lr < min_lr:
                self.lr = min_lr
            else:
                self.lr = new_lr


