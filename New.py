import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import datasets,models,transforms
from torch.autograd import Variable
from torch import nn
import time

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir=""

# dog_data_transform=transforms.Compose([transforms.Resize(224,224)],
#                                   transforms.ToTensor(),
#                                   transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]))
# cat_data_transform=transforms.Compose([transforms.Resize(224,224)],
#                                   transforms.ToTensor(),
#                                   transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]))

data_transform={x:transforms.Compose([transforms.Resize(224,224)],
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]))
                for x in ["train","valid"]}

image_datasets={x:datasets.ImageFolder(root=os.path.join(data_dir,x),transform=data_transform[x])
               for x in ["train","valid"]}

data_loader={x:torch.utils.data.DataLoader(dataset=image_datasets[x],
                                           batch_size=16,
                                           shuffle=True)
             for x in ["train","valid"]}

X_example,Y_example=next(iter(data_loader["train"]))
example_classes=image_datasets["train"].classes
index_classes=image_datasets["train"].class_to_idex

model_1=models.vgg16(pretrained=True)
for para in models.parameters:
    para.requires_gard=False
model_1.classifier=torch.nn.Sequential(
    torch.nn.Linear(25088,4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096,2)
)

model_2=models.resnet50(pretrained=True).to(device)
for para in models.parameters:
    para.requires_gard=False

model_2.fc=torch.nn.Linear(2048,2)


# 定义损失函数
loss_fun_1=torch.nn.CrossEntropyLoss()
loss_fun_2=torch.nn.CrossEntropyLoss()

# 定义优化函数
optimizer_1=torch.optim.Adam(model_1.classifier.parameters(),
                             lr=0.00001)
optimizer_2=torch.optim.Adam(model_2.fc.parameters(),
                             lr=0.00001)

weight_1=0.6
weight_2=0.4

epoch_n=5
time_open=time.time()

# 开始训练
for epoch in range(epoch_n):
    print("Epoch:{}/{}".format(epoch,epoch_n-1))
    print("-"*10)

    for phase in ["train","valid"]:
        # 进行反向传播，model的参数进行更新
        if phase=="train":
            print("Training....")
            model_1.train(True)
            model_2.train(True)

        # 不进行反向传播，model的参数不进行更新
        else:
            print("Testing....")
            model_1.train(False)
            model_2.train(False)

        running_loss_1=0.0
        running_correct_1=0
        running_loss_2=0.0
        running_correct_2=0
        blending_running_correct=0

        for batch,data in enumerate(data_loader[phase]):
            # 加载数据
            X,Y=data
            X,Y=Variable(X).to(device),Variable(Y).to(device)

            # 数据送入模型，正向传播得出结果
            Y_pred_1=model_1(X)
            Y_pred_2=model_2(X)
            blending_Y_pred=Y_pred_1*weight_1+Y_pred_2*weight_2

            # 将结果概率转化为标签
            _,pred_1=torch.max(Y_pred_1.data,1)
            _,pred_2=torch.max(Y_pred_2.data,1)
            _,blending_Y_pred=torch.max(blending_Y_pred.data,1)

            # 将每次训练时的梯度归0，共训练epoch*(len/batch_size)次
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            # 计算该次训练的损失，batch_size个数据的损失
            loss_1=loss_fun_1(Y_pred_1,pred_1)
            loss_2=loss_fun_2(Y_pred_2,pred_2)

            # 如果是训练过程，进行反向传播，梯度更新
            if phase=="train":
                loss_1.backward()
                optimizer_1.step()
                loss_2.backward()
                optimizer_2.step()

            # 计算该epoch的总损失，+=每次训练的损失
            running_loss_1+=loss_1.item()
            running_correct_1+=torch.sum(pred_1==Y.data)
            running_loss_2 += loss_2.item()
            running_correct_2 += torch.sum(pred_2 == Y.data)
            blending_running_correct+=torch.sum(blending_Y_pred==Y.data)

            if batch%500==0 and phase=="train":

                print("Batch:{},Model_1 Train Loss:{:.4f},model_1 Train Acc:{:.4f},"
                      "Model_2 Train Loss:{:.4f},model_2 Train Acc:{:.4f},"
                      "Blending_Model ACC:{:.4f}"\
                      .format(batch,running_loss_1/batch,running_correct_1/(16*batch),
                                                         running_loss_2/batch,running_correct_2/(16*batch),
                                                         blending_running_correct/(16*batch)))

            # 将损失、准确度等进行平均，计算每个样例的平均
            # loss再反向传播时会进行平均，所以这里又 *batch_size
            epoch_loss_1=running_loss_1*16/len(image_datasets[phase])
            epoch_acc_1=100*running_correct_1/len(image_datasets[phase])
            epoch_loss_2=running_loss_1*16/len(image_datasets[phase])
            epoch_acc_2=100*running_correct_1/len(image_datasets[phase])
            epoch_blending_acc=100*blending_running_correct/len(image_datasets[phase])

            print("Epoch:{},Model_1 Train Loss:{:.4f},model_1 Train Acc:{:.4f},"
                  "Model_2 Train Loss:{:.4f},model_2 Train Acc:{:.4f},"
                  "Blending_Model ACC:{:.4f}" \
                  .format(epoch, epoch_loss_1, epoch_acc_1,
                          epoch_loss_2, epoch_acc_2,
                          epoch_blending_acc))

            time_end=time.time()-time_open
            print(time_end)












