import numpy
import sys
import os
import pdb

class Dataset:
  '''Slices, shuffles and manages a small dataset for the HF optimizer.'''

  def __init__(self, data, batch_size, number_batches=None,targets=None):
    '''SequenceDataset __init__

  data : list of lists of numpy arrays
    Your dataset will be provided as a list (one list for each graph input) of
    variable-length tensors that will be used as mini-batches. Typically, each
    tensor is a sequence or a set of examples.
  batch_size : int or None
    If an int, the mini-batches will be further split in chunks of length
    `batch_size`. This is useful for slicing subsequences or provide the full
    dataset in a single tensor to be split here. All tensors in `data` must
    then have the same leading dimension.
  number_batches : int
    Number of mini-batches over which you iterate to compute a gradient or
    Gauss-Newton matrix product. If None, it will iterate over the entire dataset.
  minimum_size : int
    Reject all mini-batches that end up smaller than this length.'''
    self.current_batch = 0
    self.number_batches = number_batches
    self.items = []
    if targets == None:
      if batch_size is None:
        #self.items.append([data[i][i_sequence] for i in xrange(len(data))])
        self.items = [[data[i]] for i in xrange(len(data))]
      else:
        #self.items = [sequence[i:i+batch_size] for sequence in data for i in xrange(0, len(sequence), batch_size)]
        
        for sequence in data:
          num_batches = sequence.shape[0]/float(batch_size)
          num_batches = numpy.ceil(num_batches)
          for i in xrange(int(num_batches)):
            start = i*batch_size
            end = (i+1)*batch_size
            if end > sequence.shape[0]:
              end = sequence.shape[0]
            self.items.append([sequence[start:end]])
    else:
      if batch_size is None:
        self.items = [[data[i],targets[i]] for i in xrange(len(data))]
      else:
        for sequence,sequence_targets in zip(data,targets):
          num_batches = sequence.shape[0]/float(batch_size)
          num_batches = numpy.ceil(num_batches)
          for i in xrange(int(num_batches)):
            start = i*batch_size
            end = (i+1)*batch_size
            if end > sequence.shape[0]:
              end = sequence.shape[0]
            self.items.append([sequence[start:end],sequence_targets[start:end]])
      

    if not self.number_batches:
      self.number_batches = len(self.items)
    self.num_min_batches = len(self.items) 
    self.shuffle()
  
  def shuffle(self):
    numpy.random.shuffle(self.items)

  def iterate(self, update=True):
    for b in xrange(self.number_batches):
      yield self.items[(self.current_batch + b) % len(self.items)]
    if update: self.update()

  def update(self):
    if self.current_batch + self.number_batches >= len(self.items):
      self.shuffle()
      self.current_batch = 0
    else:
      self.current_batch += self.number_batches