import os
import yaml
import torch


def parse_configs(config_path):
	with open(config_path, 'r') as config_file:
		config = yaml.load(config_file, Loader=yaml.FullLoader)
	
	return config

def get_optimizer(optim_name, model, lr):
	optimizers = {
		'Adam': torch.optim.Adam(model.parameters(), lr = lr),
		'SGD': torch.optim.SGD(model.parameters(), lr = lr),
		'Adagrad': torch.optim.Adagrad(model.parameters(), lr = lr),
	}
	return optimizers[optim_name]

def save_delete_ckpt(step, saving_path, model=None, optimizer=None, mode='save'):
	if mode == 'save':
		torch.save(
				{
					"model": model.state_dict(),
					"optimizer": optimizer.state_dict(),
				},
				os.path.join(
					saving_path,
					"{}.pth.tar".format(step),
				),
			)
	elif mode == 'delete':
		try:
			file_path = os.path.join(saving_path,"{}.pth.tar".format(step),)
			os.remove(file_path)
			print(f"File '{file_path}' has been removed successfully.")
		except FileNotFoundError:
			print(f"File '{file_path}' not found.")
		except Exception as e:
			print(f"An error occurred: {e}")