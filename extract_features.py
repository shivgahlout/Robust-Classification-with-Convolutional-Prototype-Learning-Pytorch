import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import math
import torch.utils.data as data_utils
import torch.nn.functional as F
from Models import*
from train_utils import*
import glob
import os
import pickle





def reshape_dataset(dataset, height, width):
    new_dataset=[]
    for k in range(0,dataset.shape[0]):
      new_dataset.append(np.reshape(dataset[k], [1,height, width])) 

    return np.array(new_dataset)

class LoadDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

with open("mnist.data", "rb") as fid:
        u= pickle._Unpickler(fid)
        u.encoding = 'latin1'
        dataset = u.load()
train_x, train_y = dataset[0]
train_x=reshape_dataset(train_x,28,28)
test_x, test_y = dataset[1]
test_x=reshape_dataset(test_x,28,28)
train_num = train_x.shape[0]
test_num = test_x.shape[0]



dataset_test=LoadDataset(test_x,test_y)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)
dataset_test_len=1.0*len(dataset_test)


model=Net(2,10,2)
def copy_data(m, i, o):
        my_embedding.copy_(o)

model = torch.load('final_model.pt')
model= model.cuda()
print(model)
layer = model._modules.get('ip1')
extracted_features=[]
true_labels=[]


i=0;
for (image, label,)  in test_loader: 
    i+=1
    print('working on {} image'.format(i))
    
    my_embedding = torch.zeros(1, 2)
    def copy_data(m, i, o):
        my_embedding.copy_(o)
    h = layer.register_forward_hook(copy_data)
    image, label = Variable(image.cuda(),volatile=True), Variable(label.cuda(1))      
    test_outputs= model(image)
    h.remove()
    my_embedding=my_embedding.squeeze(0)
    my_embedding=my_embedding.detach().numpy()
    extracted_features.append(my_embedding)
    true_labels.append(label.cpu().data.numpy()[0])

    np.save('features', extracted_features)
    np.save('labels', true_labels)