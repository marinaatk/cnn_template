import argparse
import os
from tqdm import tqdm
import pandas as pd

from cnn_models.cnn1 import CNN
from dataset import CustomDataset

import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms # transformation from here
from utils.utils import get_optimizer, parse_configs, save_delete_ckpt


def get_transforms():
	train_transforms = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
	])

	test_transforms = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor()
		])

	target_transforms = transforms.Compose([
		transforms.ToTensor()
		])

	return train_transforms, test_transforms, target_transforms


def train(model, train_loader, test_loader, loss_func, optimizer, num_epochs, device, saving_path):
	"""
	Main train loop
	Args:
		model: model object (i.g. cnn1)
		train_loader, test_loader: loaders of type DataLoader
		loss_func: loss function (i.g. torch.nn.CrossEntropyLoss())
		optimizer:
		num_epochs:
		device:
	"""
	losses_log = {}
	accuracies_log = {}
	losses_log_test = {}
	accuracies_log_test = {}
	best_steps = []  # List of tuples (step, total_loss)
	max_best_steps = 3  # Maximum number of best steps to keep
	for epoch in range(num_epochs):
		epoch_loss = 0
		epoch_accuracy = 0

		progress = tqdm(train_loader)
		for data, label in progress:
			data = data.to(device)
			label = label.to(device)

			output = model(data)
			loss = loss_func(output, label)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			acc = ((output.argmax(dim=1) == label).float().mean())
			epoch_accuracy += acc/len(train_loader)
			epoch_loss += loss/len(train_loader)

			progress.set_description('Epoch : {}, train accuracy : {:.2f}, train loss : {:.2f}'.format(epoch+1, epoch_accuracy,epoch_loss))
			losses_log[epoch+1] = epoch_loss
			accuracies_log[epoch+1] = epoch_accuracy

		best_steps.append((epoch, epoch_loss))
		best_steps.sort(key=lambda x: x[1])  # Sort based on total_loss
		save_delete_ckpt(epoch, saving_path, model=model, optimizer=optimizer, mode='save')
		# Remove the worst step if the container exceeds the maximum allowed size
		if len(best_steps) > max_best_steps:
			delete_step = best_steps[-1][0]
			# DELETE CKPT
			save_delete_ckpt(delete_step, saving_path, mode='delete')
			best_steps.pop()


		with torch.no_grad():
			epoch_val_accuracy=0
			epoch_val_loss =0
			progress_test = tqdm(test_loader)
			for data, label in progress_test:
				data = data.to(device)
				label = label.to(device)

				val_output = model(data)
				val_loss = loss_func(val_output,label)


				acc = ((val_output.argmax(dim=1) == label).float().mean())
				epoch_val_accuracy += acc/ len(test_loader)
				epoch_val_loss += val_loss/ len(test_loader)

			progress_test.set_description(print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss)))
			losses_log_test[epoch+1] = epoch_val_loss
			accuracies_log_test[epoch+1] = epoch_val_accuracy
	return model, optimizer, loss, val_loss, losses_log, losses_log_test


def main(src_train_dir, src_train_csv, src_test_dir, src_test_csv, config_path, device):
	configs = parse_configs(config_path)
	batch_size = configs['data']['batch_size']
	num_epochs = configs['training']['epochs']
	learning_rate = configs['training']['learning_rate']
	optim_name = configs['training']['optimizer']
	
	saving_path = os.path.join(configs['saving']['dir'], configs['name'])
	os.makedirs(saving_path, exist_ok=True)

	train_transforms, test_transforms, target_transforms = get_transforms()

	train_dataset = CustomDataset(src_train_dir, src_train_csv, transform=train_transforms)
	test_dataset = CustomDataset(src_test_dir, src_test_csv, transform=test_transforms)
	
	train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=True)
	
	cnn = CNN().to(device)
	print(cnn)

	loss_func = torch.nn.CrossEntropyLoss()
	optimizer = get_optimizer(optim_name, cnn, learning_rate)
	# training loop
	model, optimizer, loss, val_loss, losses_log, losses_log_test = train(cnn, train_loader, test_loader, loss_func, optimizer, num_epochs, device, saving_path)
	losses_list = [{'Epoch': epoch, 'Loss': loss.item()} for epoch, loss in losses_log.items()]
	losses_list_test = [{'Epoch': epoch, 'Loss': loss.item()} for epoch, loss in losses_log_test.items()]

	pd.DataFrame(losses_list).to_csv(os.path.join(saving_path, 'train_logs.csv'), index=False)
	pd.DataFrame(losses_list_test).to_csv(os.path.join(saving_path, 'test_logs.csv'), index=False)


if __name__ == "__main__":
	"""
	This python module trains CNN model
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("src_train_dir", type=str, help="[input] path to train source directory")
	parser.add_argument("src_train_csv", type=str, help="[input] path to input file with train annotations")
	parser.add_argument("src_test_dir", type=str, help="[input] path to test source directory")
	parser.add_argument("src_test_csv", type=str, help="[input] path to input file with test annotations")
	parser.add_argument("config", type=str, help="[input] path to config yaml file")
	parser.add_argument("--device", type=str, help="[input] cpu or cuda", default='cpu')

	args = parser.parse_args()
	main(args.src_train_dir, args.src_train_csv, args.src_test_dir, args.src_test_csv, args.config, args.device)
