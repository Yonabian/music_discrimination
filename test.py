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


    # dataloader
    test_dataset = spectrogram.spectrogram(train=False,transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True)

    # define model
    net = ResNet.ResNet(fusionType='Attention')
    net.cuda()

    #loss
    criterion = nn.CrossEntropyLoss()

    # first test
    net.load_state_dict(torch.load('save_model/lr0.01.pt'))
    epochs = 30;
    loss = [];
    print('begin test..')
    for epoch in range(epochs):
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
        loss.append(test_loss)
        correct /= len(test_loader.dataset)
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #plot
    plt.plot(range(1,51),loss, 'r--', label='learning rate = 0.01')

    # 2nd test
    net.load_state_dict(torch.load('save_model/lr0.005.pt'))
    epochs = 30;
    loss = [];
    print('begin test..')
    for epoch in range(epochs):
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
        loss.append(test_loss)
        correct /= len(test_loader.dataset)
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #plot
    plt.plot(range(1,51),loss, 'b--', label='learning rate = 0.005')

    # 3rd test
    net.load_state_dict(torch.load('save_model/lr0.001.pt'))
    epochs = 30;
    loss = [];
    print('begin test..')
    for epoch in range(epochs):
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
        loss.append(test_loss)
        correct /= len(test_loader.dataset)
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #plot
    plt.plot(range(1,51),loss, 'g--', label='learning rate = 0.001')

    # 4th test
    net.load_state_dict(torch.load('save_model/lr0.0005.pt'))
    epochs = 30;
    loss = [];
    print('begin test..')
    for epoch in range(epochs):
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
        loss.append(test_loss)
        correct /= len(test_loader.dataset)
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #plot
    plt.plot(range(1,51),loss, 'y--', label='learning rate = 0.0005')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()





