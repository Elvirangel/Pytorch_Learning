import torch
import torchvision
from torchvision import transforms,models
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from PIL import Image
import copy

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform=transforms.Compose([transforms.Resize([224,224]),
                              transforms.ToTensor()])

def loadimg(path=None):
    img=Image.open(path)
    img=transform(img)
    # ??????????????????????????
    img=img.unsqueeze(0)
    return img

content_img=loadimg("E:/MY496.jpg")
content_img=Variable(content_img).to(device)
style_img=loadimg("E:/11.jpg")
style_img=Variable(style_img).to(device)


class Content_loss(torch.nn.Module):
    def __init__(self,weight,target):
        super(Content_loss,self).__init__()
        self.weight=weight
        # .datach()对变量进行锁定，不需要进行梯度
        self.target=target.detach()*weight
        # 定义损失函数
        self.loss_fn=torch.nn.MSELoss()

    def forward(self,input):
        # 计算原始图像与生成图像在L层的卷积特征的差异,作为损失
        self.loss=self.loss_fn(input*self.weight,self.target)
        # 等同于：target没有乘以weight
        # self.loss=self.loss_fn(input,target)*self.weight
        # ????????????
        return input

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss


class Style_loss(torch.nn.Module):
    def __init__(self,weight,target):
        super(Style_loss,self).__init__()
        self.weight=weight
        self.target=target.detach()*weight
        self.loss_fn=torch.nn.MSELoss()
        self.gram=Gram_matrix()

    def forward(self,input):
        self.Gram=self.gram(input.clone())
        self.Gram.mul_(self.weight)
        self.loss=self.loss_fn(self.Gram,self.target)
        return input

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss


class Gram_matrix(torch.nn.Module):
    def forward(self,input):
        a,b,c,d=input.size()
        feature=input.view(a*b,c*d)
        gram=torch.mm(feature,feature.t())
        # ??????????????????
        return gram.div(a*b*c*d)


cnn=models.vgg16(pretrained=True).features.to(device)

content_layer=["Conv_3"]
style_layer=["Conv_1","Conv_2","Conv_3""Conv_4"]

content_losses=[]
style_losses=[]

content_weight=1
style_weight=1000



new_model=torch.nn.Sequential().to(device)
# deep copy保证复杂的对象是独立的
model=copy.deepcopy(cnn)
gram=Gram_matrix().to(device)

index=1
for layer in list(model)[:8]:
    if isinstance(layer,torch.nn.Conv2d):
        name="Conv_"+str(index)
        new_model.add_module(name,layer)
        if name in content_layer:
            target=new_model(content_img).clone()
            content_loss=Content_loss(content_weight,target)
            new_model.add_module("content_loss_"+str(index),content_loss)
            content_losses.append(content_loss)

        if name in style_layer:
            target=new_model(style_img).clone()
            target=gram(target)
            style_loss=Style_loss(style_weight,target)
            new_model.add_module("style_loss_"+str(index),style_loss)
            style_losses.append(style_loss)

    if isinstance(layer,torch.nn.ReLU):
        name="Relu_"+str(index)
        new_model.add_module(name,layer)
        index=index+1

    if isinstance(layer,torch.nn.MaxPool2d):
        name="MaxPool_"+str(index)
        new_model.add_module(name,layer)

print(new_model)




input_img=content_img.clone()
parameter=torch.nn.Parameter(input_img.data)
optimizer=torch.optim.LBFGS([parameter])

epoch_n=30
epoch=[0]
while epoch[0]<=epoch_n:
    def closure():
        optimizer.zero_grad()
        style_score=0
        content_score=0
        parameter.data.clamp_(0,1)
        new_model(parameter)
        for sl in style_losses:
            style_score+=sl.backward()

        for cl in content_losses:
            content_score+=cl.backward()

        epoch[0]+=1
        if epoch[0]%5==0:
            print('Epoch:{} Style Loss:{}:4f Content Loss:{:4f}'.format(
                epoch[0],style_score.item(),content_score.item()))

        return style_score+content_score

    optimizer.step(closure)
output=parameter.data.cpu().numpy()[0].transpose([1,2,0])
plt.imshow(output)
plt.show()








