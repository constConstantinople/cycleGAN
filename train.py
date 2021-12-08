import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from model import CycleGAN
import time

class FaceDataSet:
    def __init__(self, batch_size=5, dataset_path=""):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.train_dataset = self.get_train_numpy()
        self.trans = self.get_transform()
        self.train_loader, self.val_loader = self.get_dataloader()
        
    def get_train_numpy(self):
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'train'))
        train_x = np.zeros((len(train_dataset), 224, 224, 3))
        # train_x = np.zeros((len(train_dataset), 64, 64, 3))
        for i, (img, _) in enumerate(train_dataset):
            train_x[i] = img
        return train_x / 255.0
    
    def get_transform(self):
        x_mean = np.mean(self.train_dataset, axis=(0, 1, 2))
        x_std = np.std(self.train_dataset, axis=(0, 1, 2))
        
        trans_list = [
            transforms.ToTensor(),
            transforms.Normalize(x_mean, x_std)
        ]
        
        trans = transforms.Compose(trans_list)
        return trans
        
    def get_dataloader(self):
        train_set = torchvision.datasets.ImageFolder(self.dataset_path, transform=self.trans)
        val_set = torchvision.datasets.ImageFoler(self.dataset_path, transforms=self.trans)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size)
        return train_loader, val_loader

def train(dataset, model, config):
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    G_A, G_B, D_A, D_B, cycle_A, cycle_B = [],[],[],[],[],[]
    for epoch in config.n_epochs:
        n_iter = 0
        # epoch_start = time.time()
        model.lr_update()
        for A, B in train_loader:
            A, B = A.cuda(), B.cuda()
            # iter_start = time.time()
            n_iter += config.batch_size
            model.Optimize(A, B)
            
            if n_iter % config.print_freq:
                print('print loss after %d iter' % (n_iter))
                print("G_A: %f, G_B: %f, D_A: %f, D_B: %f, cycle_A: %f, cycle_B: %f" % (model.genLoss_A,model.genLoss_B,model.disLoss_A,model.disLoss_B,model.cycleLoss_A,model.cycleLoss_B))

        if epoch % config.save_freq == 0:
            print('saving the model, epoch %d' % (epoch))
            model.model_save(epoch)
            # TO DO: print pictures from model.real_A,fake_B,rec_A,real_B,fake_A,rec_B

        G_A += model.genLoss_A
        G_B += model.genLoss_A
        D_A += model.disLoss_A
        D_B += model.disLoss_B
        cycle_A += model.cycleLoss_A
        cycle_B += model.cycleLoss_B

    plt.title('G_A')
    plt.plot(G_A)
    plt.show()
    plt.title('G_B')
    plt.plot(G_B)
    plt.show()
    plt.title('D_A')
    plt.plot(D_A)
    plt.show()
    plt.title('D_B')
    plt.plot(D_B)
    plt.show()
    plt.title('cycle_A')
    plt.plot(cycle_A)
    plt.show()
    plt.title('cycle_B')
    plt.plot(cycle_B)
    plt.show()
	
    return 
            

if __name__ == '__main__':
    
    config = {
        'save_freq': 10,
        'print_freq': 100,
        'batch_size': 10,
        'dataset_path': '',
        'is_train': True,
        'is_gpu': False,
        'checkpoints_dir': './checkpoints',
        'print_dir': './result',
        'name': 'face2comic',
        'output_nc': 3,
        'input_nc': 3,
        'n_layers': 3,
        'n_filter': 64,
        'n_epochs': 150,
        'n_epochs_decay': 150,
        'norm': 'batch',
        'init_type': 'normal',
        'init_gain': 0.02,
        'beta': 0.5,
        'epoch_count': 1,
        'lr': 0.0002,
        'lr_decay_iters': 50,
        'lr_policy': 'linear',
		'lambda_A': 10.0,
		'lambda_B': 10.0
    }
    dataSet = FaceDataSet(batch_size=config['batch_size'], dataset_path=config['dataset_path'])
    trainModel = CycleGAN(config)
    
    train(dataSet, trainModel, config)
