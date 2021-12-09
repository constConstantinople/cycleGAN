import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from model import CycleGAN
import time
from arguments import Arguments
from dataset import FaceDataSet


def train(dataset, model, config):
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    G_A, G_B, D_A, D_B, cycle_A, cycle_B = [],[],[],[],[],[]
    for epoch in range(config.n_epochs):
        n_iter = 0
        # epoch_start = time.time()
        
        for batch_data in train_loader:
            print(len(batch_data))
            A = batch_data[0]
            B = batch_data[1]
            print(type(A),A.size())
            print(type(B),B.size())
            # iter_start = time.time()
            n_iter += config.batch_size
            model.Optimize(A, B)
            
            if n_iter % config.print_freq == 0:
                print('print loss after %d iter' % (n_iter))
                print("G_A: %f, G_B: %f, D_A: %f, D_B: %f, cycle_A: %f, cycle_B: %f" % (model.genLoss_A,model.genLoss_B,model.disLoss_A,model.disLoss_B,model.cycleLoss_A,model.cycleLoss_B))

        if epoch % config.save_freq == 0:
            print('saving the model, epoch %d' % (epoch))
            model.model_save(epoch)
            # TO DO: print pictures from model.real_A,fake_B,rec_A,real_B,fake_A,rec_B

        model.lr_update()

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
    
    config = Arguments().parse()
    dataSet = FaceDataSet(batch_size=config.batch_size, dataset_path=config.dataset_path)
    trainModel = CycleGAN(config)
    
    if config.is_train:
        train(dataSet, trainModel, config)
