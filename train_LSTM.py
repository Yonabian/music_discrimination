import torch
from model import LSTMFCN
from dataset import timeseries
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':


    train_dataset = timeseries.timeseries(train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)


    test_dataset = timeseries.timeseries(train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True)

    model = LSTMFCN.LSTMFCN(N_time=20,N_Features=92,isOCT=True,isGRU=True).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    # with torch.no_grad():
    #     inputs = prepare_sequence(training_data[0][0], word_to_ix)
    #     tag_scores = model(inputs)
    #     print(tag_scores)

    for epoch in range(30): 
        model.train()
        for batch_idx, (inputEnu, labelEnu) in enumerate(train_loader):
            correct=0
            optimizer.zero_grad()
            inputs, labels = Variable(inputEnu).cuda(), Variable(labelEnu).cuda()
            tag_scores = model(inputs)
            loss = criterion(tag_scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 1 == 0:
                print("epoch{}, batch_idx{}, loss: {}".format(epoch, batch_idx, loss.item()))

        # See what the scores are after training
        model.eval()
        correct,test_loss = 0,0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.cuda(), y.cuda()
                pred = model(X)
                test_loss += criterion(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
        test_loss /= len(test_loader)
        correct /= len(test_loader.dataset)
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

