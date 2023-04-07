import numpy as np
import torch
from torch.utils import data
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import transforms
from torch import nn, optim
from matplotlib import pyplot as plt
from sklearn.preprocessing import  MinMaxScaler
from MLP_GLU import MLP_GLU
import math
# from model_c2 import Cnn1
from tqdm import tqdm, trange
import os
#CNN+MLP混合
batch_size =80
learning_rate = 0.00001
epochs =2000

flag=0
input_size=5
input_size=torch.tensor(input_size)
output_size=96
output_size=torch.tensor(output_size)
hidden_size=1024 #隐藏层的节点数量
hidden_size=torch.tensor(hidden_size)
num_layers=3#隐藏层数量
num_layers=torch.tensor(num_layers)
test_loss_select=0
Dir='Model_GLU_X_Y/'
loss_folder='Model_GLU_X_Y/loss_X_to_Y_GLU/'
model_pkl_folder='Model_GLU_X_Y/model_pkl_X_to_Y_GLU/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#device = torch.device('cpu')#('cuda:0')
print(device)
count_filename = "count_X_to_Y_GLU.txt"
# 如果文件不存在，则初始化计数器为0
if not os.path.exists(count_filename):
    with open(count_filename, "w") as f:
        f.write("0")
# 读取计数器的当前值
with open(count_filename, "r") as f:
    count = int(f.read())
# 检查 modelcc.txt 文件是否存在，如果不存在则创建
if not os.path.isfile('model_X_to_Y_GLU.txt'):
    open('model_X_to_Y_GLU.txt', 'w').close()
if count>=1:
    # 读取 model.txt 文件的最后一行，获取上次运行的 pkl 文件名
    with open('model_X_to_Y_GLU.txt', 'r') as f:
        linescc = f.readlines()
        if linescc:
            last_linecc = linescc[-1].strip()
        else:
            last_linecc = ""
l = 1 # 厚度
d = 6# 直径
fr = 18500  # 剩余磁场
u0 = torch.tensor(4 * math.pi * (10 ** -7))
pi = torch.tensor(math.pi)
m1 = (((math.pi) * d * d * l * fr) / (4 * u0))
x = np.loadtxt('DATASET/ssfin.txt', delimiter=',') # another list of numpy arrays (targets)
y = np.loadtxt('DATASET/ccfin.txt', delimiter=',') # another list of numpy arrays (targets)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
# x = np.loadtxt('d_new.txt', delimiter=',') # another list of numpy arrays (targets)
# y = np.loadtxt('l_new.txt', delimiter=',') # another list of numpy arrays (targets)
# x = np.loadtxt('d_test.txt', delimiter=',') # another list of numpy arrays (targets)
# y = np.loadtxt('l_test.txt', delimiter=',') # another list of numpy arrays (targets)
l1 = 101
# y_train=np.loadtxt('DATASET/cctrain.txt',delimiter=',')
# x_train = np.loadtxt('DATASET/sstrain.txt',delimiter=',')
# y_test=np.loadtxt('DATASET/cctest.txt',delimiter=',')
# x_test = np.loadtxt('DATASET/sstest.txt',delimiter=',')


x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)  # float32
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = data.TensorDataset(x_train, y_train)
test_dataset = data.TensorDataset(x_test, y_test)

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#shuffle是否对数据进行打乱
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
train = list(enumerate(train_dataloader))
test = list(enumerate(test_dataloader))


net_glu_96_5= MLP_GLU(input_size, output_size, hidden_size, num_layers,device).to(device) #96->5
if count>=1:
        net_glu_96_5 .load_state_dict(torch.load(last_linecc))

optimizer_c = optim.Adam(net_glu_96_5.parameters(),lr=learning_rate)#,weight_decay=0.005,weight_decay=0.5
criteon = nn.MSELoss().to(device)
#criteon =nn.SmoothL1Loss().to(device)
# x_new = x_new.T
g = 0
m = 0
losstest = np.array([])
losstrain = np.array([])
losstest_x = np.array([])
losstrain_x = np.array([])
epoch_train = np.array([])
epoch_test = np.array([])

# predict_the = torch.tensor([200, 200, 200, 200, 200, 200, 200, 200, 200]).to(device)
predict_the = torch.from_numpy(np.zeros(384)).to(device)
predict_the = predict_the.view(1, -1)
predict_the = predict_the.repeat_interleave(batch_size, dim=0)

for epoch in range(epochs):
    for batch_idx in tqdm(range(len(train)), desc='train'):
        _, (data, target) = train[batch_idx]
        data, target = data.to(device), target.to(device)
        if data.size()[0] == predict_the.size()[0]:
            # x_input = torch.cat((data, target), dim=1).float()
            #print("aaa"+str(data.device))
            output = net_glu_96_5(target,device).requires_grad_(True).to(device)
            loss_f = criteon(output,data)
            optimizer_c.zero_grad()
            loss_f.backward(retain_graph=True)
            optimizer_c.step()
            #total_loss = criteon(fin_out,target)

            losstrain = np.append(losstrain, loss_f .item())
        # 最后的batch不足batch_size时
        else:
             # x_input = torch.cat((data, target), dim=1).float()
            output = net_glu_96_5(target,device).requires_grad_(True).to(device)
            loss_f = criteon(output,data)
            optimizer_c.zero_grad()
            loss_f.backward(retain_graph=True)
            optimizer_c.step()
            #total_loss = criteon(fin_out,target)
            losstrain = np.append(losstrain, loss_f.item())
        # 最后的batch不足batch_size时


    epoch_losstrain = np.mean(losstrain)
    epoch_train = np.append(epoch_train, epoch_losstrain)
    losstrain = np.array([])

    print('Train Epoch: {}:  TotalLoss:{:6f}'.format(
        epoch, epoch_losstrain
    ))
    test_loss = 0
    for batch_idx in tqdm(range(len(test)), desc='test'):
        _, (test_data, test_target) = test[batch_idx]
        test_data, test_target = test_data.to(device), test_target.to(device)

        if test_data.size()[0] == predict_the.size()[0]:

            output = net_glu_96_5(test_target,device).requires_grad_(True).to(device)

            test_loss = criteon(output, test_data)
            losstest = np.append(losstrain, test_loss.item())
        else:
           output = net_glu_96_5(test_target,device).requires_grad_(True).to(device)

           test_loss = criteon(output, test_data)
           losstest = np.append(losstrain, test_loss.item())
    epoch_losstest = np.mean(losstest)

    epoch_test = np.append(epoch_test, epoch_losstest)
    temp=epoch_losstest
    # 读取计数器的当前值
    if(test_loss_select>temp):
        with open(count_filename, "r") as f:
            count = int(f.read())
        losstest = np.array([])
        losstest_x = np.array([])
        g = epoch_losstest
        print('当前最优解Test set :epoch :{}***Total loss:{:6f}\n'.format(
            epoch,epoch_losstest
        ))
        torch.save(net_glu_96_5.state_dict(), model_pkl_folder+'GLU_5_96'+ str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count) +'.pkl')
        test_loss_select=temp
    else:
        print('非最优解Test set :Total loss:{:6f}\n'.format(
            epoch_losstest
        ))
    if(flag==0):
        test_loss_select=epoch_losstest 
        flag=1
#     if g < 10:
#         torch.save(net_f.state_dict(), 'model_pkl/finlf'+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count)+'.pkl')
#         # 在程序运行结束后，将本次运行的 pkl 文件名追加到 model.txt 文件的末尾
#         with open('modelff.txt', 'a') as f:
#             f.write('model_pkl/finlf'+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count)+'.pkl\n')  # 将 pkl 文件名替换为你的实际文件名
#         torch.save(net_glu_96_5.state_dict(), 'model_pkl/finlc'+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count)+'.pkl')
#         with open('modelcc.txt', 'a') as f:
#             f.write('model_pkl/finlc' + str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count) + '.pkl\n')
#         print("找到较优解:", g)
#         m = 1
# if m == 1:
#     print("已经找到较优解")
# else:
#     print("没有找到较好的解")

    
with open('model_X_to_Y_GLU.txt', 'a') as f:
        f.write(model_pkl_folder+'GLU_5_96' + str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count) +'.pkl\n')

import matplotlib.pyplot as plt

with open(loss_folder+'位置误差train'+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count)+'.txt', 'a+') as f:
    np.savetxt(f,epoch_train)#np.savetxt(r'test.txt', x, fmt='%d', newline='-|-')
    f.close()
with open(loss_folder+'位置误差test'+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count)+'.txt', 'a+') as f:
    np.savetxt(f,epoch_test)#np.savetxt(r'test.txt', x, fmt='%d', newline='-|-')
    f.close()

# plot_epochs = []
# for i in range(epochs):
#     plot_epochs.append(int(i))
# plot_epochs = np.array(plot_epochs)
# plt.plot(plot_epochs, epoch_train, color="b")
# plt.plot(plot_epochs, epoch_test, color="r")
# # plt.xlim((0, epochs-1))
# # plt.ylim((0, 5000))
# plt.show()

#绘制训练集和测试集loss下降曲线
plot_epochs = []
for i in range(epochs):
    plot_epochs.append(int(i))
plot_epochs = np.array(plot_epochs)
plt.title('Position Loss') 
plt.plot(plot_epochs,epoch_train, label='Training Loss',color='b')
plt.plot(plot_epochs, epoch_test,label='Test Loss',color='r')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(loss_folder+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count)+'.jpg',dpi=200)
plt.close()


count += 1
with open(count_filename, "w") as f:
    f.write(str(count))