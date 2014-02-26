"""
Class that preprocesses the data
Siddharth Sigtia
Feb,2014
C4DM
"""
import numpy
import os
import time
import cPickle as pickle
import sklearn.preprocessing as preprocessing
import tables as T
import sys
import os
from utils import *
import pdb

class PreProcessor():
    def __init__(self,dataset_dir,):
        self.dataset_dir = dataset_dir
        self.feat_dir = os.path.join(self.dataset_dir,'features')
        self.list_dir = os.path.join(self.dataset_dir,'lists')
        self.h5_filename = os.path.join(self.feat_dir,'feats.h5')
        self.ground_truth = pickle.load(open(os.path.join(self.list_dir,'ground_truth.pickle'),'r'))
        self.load_data()

    def load_h5(self,):
        print 'Loading data from %s'%(self.h5_filename)
        with T.openFile(self.h5_filename,'r') as f:
            feats = f.root.x.read()
            filenames = f.root.filenames.read()
        return feats,filenames

    def load_data(self,):
        features,filenames = self.load_h5()
        self.initial_shape = features.shape[1:]
        self.n_per_example = numpy.prod(features.shape[1:-1])
        self.n_features = features.shape[-1]
        self.features,self.filenames = self.flatten_data(features,filenames)
        self.preprocess(self.features)
        self.filedict = self.build_file_idx_dict()

    def flatten_data(self,data,targets):
        flat_data = data.view() #Check if reshape is more efficient
        flat_data.shape = (-1,self.n_features)
        flat_targets = targets.repeat(self.n_per_example)
        return flat_data,flat_targets

    def preprocess(self,data,scale=True,new_sigma=None,new_mean=None):
        print 'Preprocesssing data...'
        if scale:
            self.scaler = preprocessing.StandardScaler().fit(data)
            self.scaler.transform(data)
        if new_sigma:
            data/=new_sigmase
        if new_mean:
            data += new_mean

    def unflatten_data(self,flat_data,flat_targets):
        new_shape = (-1,) + self.initial_shape
        data = flat_data.reshape*(new_shape)
        targets = flat_targets[::self.n_per_example]
        return data,targets

    def build_file_idx_dict(self,):
        keys = list(set(self.filenames))
        file_dict = dict([(key,[]) for key in keys])
        for i,filename in enumerate(self.filenames):
            file_dict[filename].append(i)
        for k in keys:
            file_dict[k] = numpy.array(file_dict[k])
        return file_dict

    def prepare_subdataset(self,split='train',unflatten=False,randomize=False):
        ids = []
        for filename in self.lists[split]:
            id = self.filedict[filename]
            ids.extend(id)
        data = self.features[ids]
        files = self.filenames[ids]
        targets = self.make_targets(files)
        if unflatten:
            data,targets = self.unflatten_data(data,targets)
        if randomize:
            data,targets = self.randomize(data,targets)
        self.data[split] = data
        self.targets[split] = targets

    def prepare_fold(self,train_list_file,valid_list_file,test_list_file):
        self.lists = {}
        self.data = {}
        self.targets = {}
        [train_list,valid_list,test_list] = self.get_fold_lists(train_list_file,valid_list_file,test_list_file)
        self.lists['train'] = train_list
        self.lists['valid'] = valid_list
        self.lists['test'] = test_list
        self.prepare_subdataset('train')
        self.prepare_subdataset('valid')
        self.prepare_subdataset('test')

    def make_targets(self,filenames):
    	targets = []
    	for filename in filenames:
    		targets.append(self.ground_truth[filename])
    	return numpy.array(targets)

    def get_fold_lists(self, train_list_file,valid_list_file,test_list_file):
        return [self.parse_list(train_list_file),
                self.parse_list(valid_list_file),
                self.parse_list(test_list_file)]

    def parse_list(self, list_file):
        if list_file is not None:
            return list(set([line.strip().split('\t')[0] for line in open(list_file,'r').readlines()]))
