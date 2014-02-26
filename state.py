"""
Parameters

MLP Parameters

n_inputs
n_outputs
n_hidden:list of number hidden layers, eg: [50,50,50]
activaion: string with activation name. Current options sigmoid or ReLU
output_layer: String containing type of output layer. Current options softmax or sigmoid
dropout_rates: List containing dropout rates for each layer. Applies dropout to all 
			   layers except the output layer. 

SGD Parameters

params: List of shared variables that are to be optimized by SGD. 
costs: List of costs that used during optimization. 1st element of the list MUST be the 
		objective function that is optimized. 
momentum: bool, tells the optimizer if it should use momentum
learning_rate: learning_rate
num_epochs: Number of epochs for SGD.
save: bool, saves the best models if True
output_folder: String, saves the best models to specified output_folder
lr_update:bool, updates learning rate if True

Trainer Parameters

dataset_dir: String, path to dataset directory.
batch_size: Minibatch size used for SGD.


"""

def get_state():
	state = {}
	state['n_inputs'] = 513
	state['n_outputs'] = 10
	state['n_hidden'] = [50,50,50]
	state['activation'] = 'sigmoid'#'ReLU'
	state['output_layer'] = 'sigmoid'#softmax
	state['dropout_rates'] = [0,0.5,0.5]
	state['momentum'] = False
	state['learning_rate'] = 0.1
	state['num_epochs'] = 200
	state['save'] = True
	state['output_folder'] = None
	state['lr_update'] = False
	state['dataset_dir'] = '/homes/sss31/datasets/gtzan' #/path/to/dataset/dir
	state['batch_size'] = 20
	state['mom_rate'] = 0.9
	return state
