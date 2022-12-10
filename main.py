import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from low_precision_utils import SConv2d, SLinear
from torch.autograd import Variable
from converter import MyFloat
import easydict

# argument parser
import easydict
 
args = easydict.EasyDict({
        "batch_size": 32,
        "epochs": 100,
        "lr": 0.001,
        "enable_cuda" : True,
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
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
)
test_set = dsets.FashionMNIST(
    root = './data/FashionMNIST',
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

class MyConvNet(nn.Module):
    def __init__(self, args):
        super(MyConvNet, self).__init__()
        self.conv1 = SConv2d(1, 16, kernel_size=3, stride=1, 
                               padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.act1  = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = SConv2d(16, 32, kernel_size=3, stride=1, 
                               padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.act2  = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.lin2  = SLinear(mf, 7*7*32, 10)

    def forward(self, x):
        c1 = self.conv1(x)
        b1  = self.bn1(c1)
        a1  = self.act1(b1)
        p1  = self.pool1(a1)
        c2  = self.conv2(p1)
        b2  = self.bn2(c2)
        a2  = self.act2(b2)
        p2  = self.pool2(a2)
        flt = p2.view(p2.size(0), -1)
        out = self.lin2(flt)
        return out
  
model = MyConvNet(args)
model = model.cuda() 

criterion = nn.CrossEntropyLoss()
criterion=criterion.cuda() 
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data=data.cuda() 
        target=target.cuda() 
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data=data.cuda() 
        target=target.cuda() 
        output = model(data)
        test_loss += criterion(output, target).data[0]
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