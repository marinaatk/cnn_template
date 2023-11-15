# This is a Convolutional Neural Network template written in PyTorch

This package runs training process for CNN in PyTorch.

Inputs:
```
dataset/
	...
	train_set/
	test_set/
```

Outputs:
```
saved_models/
	cnn1/
	cnn2/
	...
```

In order to run:
`python3 main.py dataset/training_set/ dataset/training_set/train.csv dataset/test_set/ dataset/test_set/test.csv config/cnn1.yaml --device cuda`

```
usage: main.py [-h] [--device DEVICE] src_train_dir src_train_csv src_test_dir src_test_csv config

positional arguments:
  src_train_dir    [input] path to train source directory
  src_train_csv    [input] path to input file with train annotations
  src_test_dir     [input] path to test source directory
  src_test_csv     [input] path to input file with test annotations
  config           [input] path to config yaml file

optional arguments:
  -h, --help       show this help message and exit
  --device DEVICE  [input] cpu or cuda
```