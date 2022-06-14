import torch
import torch.nn as nn
from model import ResNet
from dataset import spectrogram
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

if __name__ == '__main__':
    # resume 
    # resume_checkpoint_path = 'model.pt'

    # dataloader
    train_dataset = spectrogram.spectrogram(train=True,transform=transforms.ToTensor())
    

    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    

    # 1
    test_dataset = spectrogram.spectrogram(train=False,transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_dataset, batch_size=6, shuffle=True)
    # define model
    net = ResNet.ResNet(fusionType='Attention',r=8)
    net.cuda()

    #optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    # optim.load_state_dict(checkpoint['optim_state_dict'])

    #loss
    criterion = nn.CrossEntropyLoss()

    epochs = 30;
    print('begin train...')
    for epoch in range(epochs):
        net.train()
        # train
        # ts = time.time()
        for batch_idx, (inputEnu, labelEnu) in enumerate(train_loader):
            correct=0
            # train loop
            optimizer.zero_grad()
            inputs, labels = Variable(inputEnu).cuda(), Variable(labelEnu).cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 1 == 0:
                print("epoch{}, batch_idx{}, loss: {}".format(epoch, batch_idx, loss.item()))
        # test
        net.eval()
        correct,test_loss = 0,0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.cuda(), y.cuda()
                pred = net(X)
                test_loss += criterion(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
        test_loss /= len(test_loader)
        correct /= len(test_loader.dataset)
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        # save model
        # print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        # torch.save(net,'save_model/pop.pt')

