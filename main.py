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
import torch.utils.data as utils
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='initial_learning_rate')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--h', type=int, default=2, help='dimension of the hidden layer')
    parser.add_argument('--scale', type=float, default=2, help='scaling factor for distance')
    parser.add_argument('--reg', type=float, default=.001, help='regularization coefficient')

    args, _ = parser.parse_known_args()

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




    dataset_train=LoadDataset(train_x,train_y)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=1)

    dataset_test=LoadDataset(test_x,test_y)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=1)

    dataset_test_len=1.0*len(dataset_test)
    dataset_train_len=1.0*len(dataset_train)


    model=Net(args.h, args.num_classes, args.scale)
    model = model.cuda()

    lrate=args.lr  
    optimizer_s = optim.SGD(model.parameters(), lr=lrate, momentum=0.9, weight_decay=1e-4)

    num_epochs = 30


    plotsFileName='./plots/mnist+' #Filename to save plots. Three plots are updated with each epoch; Accuracy, Loss and Error Rate
    csvFileName='./stats/mnist_log.csv' #Filename to save training log. Updated with each epoch, contains Accuracy, Loss and Error Rate

    print(model)

    train_model(model, optimizer_s,lrate,num_epochs,args.reg, train_loader,test_loader,dataset_train_len, dataset_test_len,plotsFileName,csvFileName)