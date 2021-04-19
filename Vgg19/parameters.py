import torch.nn as nn

data_path='/home/dhia/data/train'
test_path='/home/dhia/data/test'
IMG_SIZE = 224    # Data transform size to (IMG_SIZE,IMG_SIZE)
batch_size = 10   # Loader Batch size

#torch parameters

#i add this to be able to use cuda when u have it
## run the following to know if u can use cuda
### dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev='cpu'
#learning parameters
epochs = 10
criteria= nn.CrossEntropyLoss()
learning_rate=0.001
