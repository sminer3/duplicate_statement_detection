import numpy as np
import json


#NN constant parameters
nn_constants = {'emb_shape': 300,'features_shape': 24}

#NN possible hyperparameters (integer)
nn_int = {'layer1': [25,500], 'layer2':[25,300], 'features': [30,35]}

#NN possible hyperparameters (continuous)
nn_continuous = {'dropout_rate1': [0,.5], 'dropout_rate2': [0,.5], 'dropout_rate_features': [0,.5]}

#NN possible hyperparameters (log continuous)
nn_log = {'layer1_reg': [.00001, .1], 'layer2_reg': [.00001, .01], 'feature_reg': [.00001, .1]}

#NN possible hyperparameters (discrete)
nn_discrete = {'n_layer1_hidden': [0,1], 'n_layer2_hidden': [0,1], 'activation': ['elu','relu']}

#Other param
other_discrete = {'batch_size': [32, 64, 128, 256, 512, 1024], 'optimizer': ['nadam','adam']}
other_int = {'epochs': [5,20]}
other_log = {'learning_rate': [.00001,.1]}

settings = []
for i in range(60):
	#NN hyper parameters
	nn_dict = {}
	nn_dict.update(nn_constants)
	nn_to_add = {}
	for key, value in nn_int.items():
		nn_to_add[key] = np.random.randint(value[0],value[1]+1)
	nn_dict.update(nn_to_add)
	nn_to_add = {}
	for key, value in nn_continuous.items():
		nn_to_add[key] = np.random.uniform(value[0],value[1])
	nn_dict.update(nn_to_add)
	nn_to_add = {}
	log_values = np.log10([.00001,.1])
	reg_value = 10**np.random.uniform(log_values[0],log_values[1])
	for key, value in nn_log.items():
		nn_to_add[key] = reg_value
	nn_dict.update(nn_to_add)
	nn_to_add = {}
	for key, value in nn_discrete.items():
		nn_to_add[key] = value[np.random.randint(0,len(value))]
	nn_dict.update(nn_to_add)

	dict_all = {'nn_dict': nn_dict}
	for key, value in other_discrete.items():
		other_to_add = {}
		other_to_add[key] = value[np.random.randint(0,len(value))]
		dict_all.update(other_to_add)
	for key, value in other_int.items():
		other_to_add = {}
		other_to_add[key] = np.random.randint(value[0],value[1]+1)
		dict_all.update(other_to_add)
	for key, value in other_log.items():
		other_to_add = {}
		log_values = np.log10(value)
		other_to_add[key] = 10**np.random.uniform(log_values[0],log_values[1])
		dict_all.update(other_to_add)
	
	settings.append(dict_all)

#Save configurations
with open('hyperparameters.json', 'w') as fp:
    json.dump(settings, fp)