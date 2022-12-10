import torch
import torch.nn as nn 
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from low_precision_utils_truncate import * 
import easydict
from converter import MyFloat

class SimpleNet(nn.Module):
    def __init__(self, args):
        super(SimpleNet, self).__init__()
        self.conv1 = SConv2d(1, 16, kernel_size=3, stride=1, 
                               padding=1)
        self.bn1   = SBatchNorm(16)
        self.act1  = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = SConv2d(16, 32, kernel_size=3, stride=1, 
                               padding=1)
        self.bn2   = SBatchNorm(32)
        self.act2  = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.lin2  = SLinear(7*7*32, 10)

    def forward(self, x):
        c1 = self.conv1(x)
        # print(c1.shape) 
        b1  = self.bn1(c1)
        # print(b1.shape) 
        a1  = self.act1(b1)
        # print(a1.shape) 
        p1  = self.pool1(a1)
        # print(p1.shape) 
        c2  = self.conv2(p1)
        # print(c2.shape) 
        b2  = self.bn2(c2)
        # print(b2.shape) 
        a2  = self.act2(b2)
        # print(a2.shape) 
        p2  = self.pool2(a2)
        # print(p2.shape) 
        flt = p2.view(p2.size(0), -1)
        # print(flt.shape) 
        out = self.lin2(flt)
        return out


class NetRunner():
    '''
    Trains a network to completion.
    Call run() to start training

    cmd_args is a dict with required
    '''
    def __init__(self, cmd_args={}):
        print('Arguments:', cmd_args)
        self.args = easydict.EasyDict({
            "batch_size": 32,
            "epochs": cmd_args.get('epochs', 30),   # default epochs=30
            "lr": cmd_args.get('lr', 0.1),   # default lr=0.1
            "enable_cuda" : cmd_args.get('enable_cuda', False),
            "L1norm" : False,
            "simpleNet" : True,
            "activation" : "relu", #relu, tanh, sigmoid
            "train_curve" : True, 
            "optimization" :"SGD",
            "exponent" : cmd_args.get('exponent', 5),
            "mantissa" : cmd_args.get('mantissa', 10)
        })

        # Hyper Parameters
        self.input_size = 784
        self.num_classes = 10
        self.num_epochs = self.args.epochs
        self.batch_size = self.args.batch_size
        self.learning_rate = self.args.lr
        self.loss_curve = []
        self.train_curve = []
        self.test_curve = []

        self.mf = MyFloat(
            exp_bits=self.args.exponent, 
            mant_bits=self.args.mantissa,
            device='cuda' if self.args.enable_cuda else 'cpu'
        )
        # attach float conversion to Tensor class
        torch.Tensor.mf_truncate_ = self.mf.truncate_tensor

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
        self.train_loader = torch.utils.data.DataLoader(dataset = train_set,
                batch_size = self.batch_size,
                shuffle = True)

        self.test_loader = torch.utils.data.DataLoader(dataset = test_set,
                batch_size = self.batch_size,
                shuffle = False)
        
        self.model = SimpleNet(self.args) 
        if self.args.enable_cuda:
            self.model = self.model.cuda()

        self.criterion = nn.CrossEntropyLoss()
        if self.args.enable_cuda:
            self.criterion = self.criterion.cuda() 
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.learning_rate)

    def run(self):
        for epoch in range(self.num_epochs):
            self.__train(epoch)
            self.__test()

        filename = ''
        for a in self.args:
            filename += str(a) + '_' + str(self.args[a]) + '_'
        filename = filename[:-1] + '.txt'

        with open(filename, 'w') as f:
            f.write('Arguments: ' + str(self.args) + '\n')
            f.write('Loss Curve: ' + str(self.loss_curve) + '\n')
            f.write('Train Curve: ' + str(self.train_curve) + '\n')
            f.write('Test Curve: ' + str(self.test_curve) + '\n')
            f.write('Final Test Accuracy: ' + str(self.test_curve[-1]) + '\n')



    def __train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = Variable(data), Variable(target)
            if self.args.enable_cuda:
                data=data.cuda() 
                target=target.cuda() 
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.data.item()))

            if(batch_idx == len(self.train_loader)-1):
                # save loss curve
                self.loss_curve.append(loss.data.item())
                # save train accuracy curve
                pred = output.data.max(1, keepdim=True)[1]
                correct = pred.eq(target.data.view_as(pred)).cpu().sum()
                self.train_curve.append(correct / len(data))


    def __test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            if self.args.enable_cuda:
                data=data.cuda() 
                target=target.cuda() 
            output = self.model(data)
            test_loss += self.criterion(output, target).data.item() 
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(self.test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

        # save test accuracy curve
        self.test_curve.append(correct / len(self.test_loader.dataset))
        