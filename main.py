import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from low_precision_utils_truncate import SConv2d, SLinear
from torch.autograd import Variable
from converter import MyFloat
import easydict
from SimpleNet import SimpleNet 

# argument parser
import easydict
 
args = easydict.EasyDict({
        "batch_size": 32,
        "epochs": 100,
        "lr": 0.1,
        "enable_cuda" : False,
        "L1norm" : False,
        "simpleNet" : True,
        "activation" : "relu", #relu, tanh, sigmoid
        "train_curve" : True, 
        "optimization" :"SGD",
        "exponent" : 5,
        "mantissa" : 10

})
# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
mf = MyFloat(args.exponent, args.mantissa)


# MNIST Dataset (Images and Labels)
train_set = dsets.FashionMNIST(
    root = 'FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
)
test_set = dsets.FashionMNIST(
    root = 'FashionMNIST',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
)


# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset = train_set,
        batch_size = batch_size,
        shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_set,
        batch_size = batch_size,
        shuffle = False)
  
model = SimpleNet(args) 
# model = model.cuda() 

criterion = nn.CrossEntropyLoss()
# criterion=criterion.cuda() 
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        # data=data.cuda() 
        # target=target.cuda() 
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        # data=data.cuda() 
        # target=target.cuda() 
        output = model(data)
        test_loss += criterion(output, target).data.item() 
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    for epoch in range(1, num_epochs + 1):
        train(epoch)
        test()


if __name__ == '__main__':
    main()
    print('Finished Training') 