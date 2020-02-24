import torch
import numpy as np
from torchvision import datasets,transforms
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt


# 数据转换Transform
transform=transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])])

# 下载数据集
train_data=datasets.MNIST(
    root="./data/",
    transform=transform,
    train=True,
    download=False)
test_data=datasets.MNIST(
    root="./data/",
    transform=transform,
    train=False,
    download=False)

# 数据装载
train_data_loader=torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True)
test_data_loader=torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=True)

#预览一个包的数据
images,labels=next(iter(train_data_loader))
imgs=torchvision.utils.make_grid(images)

imgs=imgs.numpy().transpose(1,2,0)
std=[0.5,0.5,0.5]
mean=[0.5,0.5,0.5]
imgs=imgs*std+mean

print(labels)
plt.imshow(imgs)


class Model(torch.nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, (3, 3), stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2,kernel_size = 2))

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14*14*128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 10))

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = x.view(-1,14*14*128)
        x = self.dense(x)
        return x

model=Model(1,64)
cost=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

epoch_n = 5

for epoch in range(epoch_n):
    running_loss = 0.0
    running_accu = 0.0
    print("Epoch:{}/{}".format(epoch, epoch_n))
    print("-" * 10)

    for data in train_data_loader:
        X_train, Y_train = data
        X_train, Y_train = Variable(X_train), Variable(Y_train)
        outputs = model(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, Y_train)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_accu += torch.sum(pred == Y_train.data)

    test_accu = 0.0
    test_loss = 0.0
    for data in train_data_loader:
        X_test, Y_test = data
        X_test, Y_test = Variable(X_test), Variable(Y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        loss = cost(outputs, Y_train)

        test_loss += loss.data[0]
        test_accu += torch.sum(pred == Y_train.data)
    print("Train_loss:{},Train_accu:{}\n,Test_loss:{},Test_accu:{}" \
          .format(running_loss, running_accu, test_loss, test_accu))

