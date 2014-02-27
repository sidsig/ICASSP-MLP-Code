This repository containts code for recreating the results in the paper "Improved Music Feature Learning with Deep Neural Networks". All the code is in Python. 
Some of the dependencies are mentioned below:
```
PyTables: For storing the extracted features
Theano: For the neural network training
Sklearn: For the pre-processing module and random forest classification
Numpy, Scipy
```

The code assumes a directory structure as follows:

```
	dataset_dir
    	|
	   audio 
```

The dataset_dir initially containts only the audio. The first step in setting up the code is to create lists for training/validation/test data. This is done by running:
```
python make_lists.py /path/to/dataset_dir
```
The above command creates the following structure:

```
	dataset_dir
       |
 lists    audio
```
The next step is to extract features (spectrograms) from the raw audio data. This is done by running the following script:
```
python feature_extraction.py /path/to/dataset_dir
```
The above command extracts features and stores them in an HDF5 file called feats.h5. The directory structure at this point looks like:
```
	dataset_dir
       |
 lists audio features
```

Most of the things are not set up to perform training. All the parameters required for training can be set in the states.py file. Training can then be initiated by calling:
```
python trainer.py
```