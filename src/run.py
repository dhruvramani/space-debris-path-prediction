import os
import gc
import csv
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import *
from dataset import *
from utils import progress_bar

# TODO : Change model
# TODO : Change dataset
# TODO : Add metric
# TODO : Change test

parser = argparse.ArgumentParser(description='PyTorch Space Debris Path Predictions')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate') 
parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--resume', '-r', type=int, default=0, help='resume from checkpoint')
parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--step_size', '-st', type=int, default=5, help='step size for rnn')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, tsepoch, tstep = 0., 0, 0

criterion = torch.nn.CrossEntropyLoss()
mse = torch.nn.MSELoss()

'''def collate_fn(data):
    data = list(filter(lambda x: type(x[1]) != int, data))
    audios, captions = zip(*data)
    data = None
    del data
    audios = torch.stack(audios, 0)
    return audios, captions'''

print('==> Creating network..')
# TODO : Change model here
net = AttentionModel(args.batch_size, [3, 3, 4], 25, args.embedding_length)
net = net.to(device)

if(args.resume):
    if(os.path.isfile('../save/network.ckpt')):
        net.load_state_dict(torch.load('../save/network.ckpt'))
        print('==> Network : loaded')

    if(os.path.isfile("../save/info.txt")):
        with open("../save/info.txt", "r") as f:
            tsepoch, tstep = (int(i) for i in str(f.read()).split(" "))
        print("=> Network : prev epoch found")
else :
    with open("../save/logs/train_loss.log", "w+") as f:
        pass 


def train_network(epoch):
    global tstep
    global best_acc
    print('\n=> Epoch: {}'.format(epoch))
    net.train()
    
    print('==> Preparing data..')
    # TODO : Change dataset
    dataset = OffenseEval(path='/home/nevronas/Projects/Personal-Projects/Dhruv/OffensEval/dataset/train-v1/offenseval-training-v1.tsv')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) #, collate_fn=collate_fn)
    dataloader = iter(dataloader)

    train_loss, accu1 = 0.0, 0.0
    le = len(dataloader)
    params = net.parameters()     
    optimizer = torch.optim.Adam(params, lr=args.lr) 

    for i in range(tstep, le):
        contents = next(dataloader)
        inputs = contents[0].type(torch.FloatTensor).to(device)
        targets = contents[1].type(torch.LongTensor).to(device)  # TODO : Change here

        optimizer.zero_grad()
        y_preds = net(inputs)
        
        loss = criterion(y_preds, targets) # TODO : Change
        tl = loss.item()
        loss.backward()
        optimizer.step()

        #acc1 =
        #accu1 += acc1 
        train_loss += tl

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(net.state_dict(), '../save/network.ckpt')
        with open("../save/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, i))

        with open("../save/logs/train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(tl))

        progress_bar(i, len(dataloader), 'Loss: {}'.format(tl)) #, acc1))

    tstep = 0
    del dataloader
    print('=> Network : Epoch [{}/{}], Loss:{:.4f}'.format(epoch + 1, args.epochs, train_loss / le))#, accu1 / le))

    #old_best = best_acc
    #best_acc = max(best_acc, accu1/le)
    #if(best_acc != old_best):
    #    torch.save(net.state_dict(), '../save/best.ckpt')
    #print("Best Metrics : {}".format(best_acc))

def test():
    global net
    net.load_state_dict(torch.load('../save/network.ckpt'))
    
    # TODO : Change data here
    dataset = OffenseEval(path='/home/nevronas/Projects/Personal-Projects/Dhruv/OffensEval/dataset/testset-taska.tsv', train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size) #, collate_fn=collate_fn)
    dataloader = iter(dataloader)

    with open('../save/test.tsv', 'w+') as f:
        for i in range(0, len(dataloader) - 1):
            contents = next(dataloader)
            inputs = contents[0].type(torch.FloatTensor).to(device)
            y_preds = net(inputs)
            # TODO : Add here

for epoch in range(tsepoch, tsepoch + args.epochs):
    train_network(epoch)

#test()