import torch.nn as nn

data_path='/home/dhia/data'
test_path='/home/dhia/test'
IMG_SIZE = 224
batch_size = 32

#torch parameters

#i add this to be able to use cuda when u have it
## run the following to know if u can use cuda
### dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 10
dev='cpu'
criteria= nn.CrossEntropyLoss()
learning_rate=0.001
