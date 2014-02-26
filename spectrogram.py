"""
Calculate the spectrogram of an audioFile
Philippe Hamel
Feb 2010
"""


import numpy as N
verbose = False

def closestPower(n):
    """
    Finds the smallest power of 2 that is >= n
    To optimize FFTs, array lengths should be powers of 2
    """
    power=1
    while (n>power):
        power*=2
    return power


class SpecGram():
    """
    A Spectrogram of an audio sample
    x : audio signal
    fs : sample rate
    winSize : lenght of fft windows (in samples)
    nOverlaps : number of overlaping windows
    """
    def __init__(self, x, fs, winSize, nOverlaps=2, logAmplitude=False, do_compress=True, window_type='square'):
        
        # let's make sure windows are a power of 2
        self.fs=fs
        self.audioLen=len(x)
        self.nOverlaps=nOverlaps
        self.winSize=closestPower(winSize)
        self.winLen=1.0*self.winSize/fs
        self.overLen=self.winSize/self.nOverlaps
        self.name='SpecGram_%i_%i'%(self.winSize,self.fs)
        self.window_type = window_type
        self.set_window()
        
        #Make sure x is in the right shape
        if x.ndim > 1:
            if x.shape[1]>1:
                print x.shape
                print 'making mono'
                x=x.mean(axis=1)
                print x.shape
        x=N.reshape(x,x.size)

#        self.numWindows=(int(len(x)/self.winSize)-1)*self.nOverlaps+1 #This always gave odd number of windows
        self.numWindows = int((len(x) - (self.winSize-self.overLen))/self.overLen)


        self.fftLen=self.winSize/2+1 #the first and middle value of the fft are not symetric
        self.nyquist=self.fs/2
        self.freqs=1.0*N.arange(self.fftLen)/(self.fftLen-1)*self.nyquist
        
        self.specMat=N.zeros((self.numWindows,self.fftLen),dtype='complex')
        self.midSample=[]
        for i in range(self.numWindows):
            self.midSample.append(i*self.overLen+self.winSize/2) 
            specData=N.fft.fft(x[i*self.overLen:i*self.overLen+self.winSize] * self.window)
            self.specMat[i,:]=specData[:self.fftLen]
        if do_compress:
            self.compress()
        if logAmplitude:
            self.specMat=N.log(self.specMat + 1e-8)
    
    def compress(self):
        """
        throws away phase and some bits
        """
        self.specMat=N.abs(self.specMat).astype(N.float32)
        
    def set_window(self):
        self.window = get_window(self.window_type, self.winSize)


def get_window(window_type, winSize):
    window_dict = {'square':square_window,
                   'triangular':triangular_window,
                   'hanning':hanning_window,
                   'hamming':hamming_window}
    window = window_dict.get(window_type,None)(winSize)        
    if window is None:
        raise ValueError, 'Unknown window: %s' % window_type
    return window
        
def square_window(winSize):
    return N.ones(winSize)

def hanning_window(winSize):
    window = N.hanning(winSize) 
    return window * winSize / window.sum()

def hamming_window(winSize):
    window = N.hamming(winSize)
    return window * winSize / window.sum()

def triangular_window(winSize):
    n = winSize
    half = (n+2)//2
    half_tri = 1.*N.arange(half)

    if n%2 == 0:
        tri= N.hstack((half_tri[1:], half_tri[1:][::-1]))
    else:
        tri= N.hstack((half_tri[1:], N.array([1.]), half_tri[1:][::-1]))
        
    return tri * winSize / tri.sum()

def spec_diff(spec1, spec2):
    ''' Computes some error on spectrograms. '''
    return float(N.log(N.mean((spec1-spec2)**2)))