"""
Feature extraction.
Siddharth Sigia
Feb,2014
C4DM
"""
import numpy
import subprocess
import sys
import os
from spectrogram import SpecGram
import tables
import pdb

def read_wav(filename):
    bits_per_sample = '16'
    cmd = ['sox',filename,'-t','raw','-e','unsigned-integer','-L','-c','1','-b',bits_per_sample,'-','pad','0','30.0','rate','22050.0','trim','0','30.0']
    cmd = ' '.join(cmd)
    print cmd
    raw_audio = numpy.fromstring(subprocess.Popen(cmd,stdout=subprocess.PIPE,shell=True).communicate()[0],dtype='uint16')
    max_amp = 2.**(int(bits_per_sample)-1)
    raw_audio = (raw_audio- max_amp)/max_amp
    return raw_audio

def calc_specgram(x,fs,winSize,):
    spec = SpecGram(x,fs,winSize)
    return spec.specMat


def make_4tensor(x):
    assert x.ndim <= 4
    while x.ndim < 4:
        x = numpy.expand_dims(x,0)
    return x

class FeatExtraction():
    def __init__(self,dataset_dir):
    	self.dataset_dir = dataset_dir
        self.list_dir = os.path.join(self.dataset_dir,'lists')
        self.get_filenames()
        self.feat_dir = os.path.join(self.dataset_dir,'features')
        self.make_feat_dir()
        self.h5_filename = os.path.join(self.feat_dir,'feats.h5')
        self.make_h5()
        self.setup_h5()
        self.extract_features()
        self.close_h5()


    def get_filenames(self,):
        dataset_files = os.path.join(self.list_dir,'audio_files.txt')
        self.filenames = [l.strip() for l in open(dataset_files,'r').readlines()]
        self.num_files = len(self.filenames)

    def make_feat_dir(self,):
    	if not os.path.exists(self.feat_dir):
    		print 'Making output dir.'
    		os.mkdir(self.feat_dir)
    	else:
    		print 'Output dir already exists.'
    
    def make_h5(self,):
    	if not os.path.exists(self.h5_filename):
    		self.h5 = tables.openFile(self.h5_filename,'w')
    	else:
    		print 'Feature file already exists.'
    		self.h5 = tables.openFile(self.h5_filename,'a')

    def setup_h5(self,):
    	filename = self.filenames[0]
    	x = read_wav(filename)
    	spec_x = calc_specgram(x,22050,1024)
    	spec_x = make_4tensor(spec_x)
    	self.data_shape = spec_x.shape[1:]
    	self.x_earray_shape = (0,) + self.data_shape
    	self.chunkshape = (1,) + self.data_shape
    	self.h5_x = self.h5.createEArray('/','x',tables.FloatAtom(itemsize=4),self.x_earray_shape,chunkshape=self.chunkshape,expectedrows=self.num_files)
    	self.h5_filenames = self.h5.createEArray('/','filenames',tables.StringAtom(256),(0,),expectedrows=self.num_files)
    	self.h5_x.append(spec_x)
    	self.h5_filenames.append([filename])


    def extract_features(self,):
        for i in xrange(1,self.num_files):
    	    filename = self.filenames[i]
            print 'Filename: ',filename
    	    x = read_wav(filename)
    	    spec_x = calc_specgram(x,22050,1024)
    	    spec_x = make_4tensor(spec_x)
    	    self.h5_x.append(spec_x)
    	    self.h5_filenames.append([filename])

    def close_h5(self,):
        self.h5.flush()
        self.h5.close()
        
if __name__ == '__main__':
	test = FeatExtraction('/homes/sss31/datasets/gtzan/')
  
