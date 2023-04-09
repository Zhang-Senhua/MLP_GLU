import torch
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import transforms
from torch import nn, optim
from matplotlib import pyplot as plt
from sklearn.preprocessing import  MinMaxScaler
from MLP_Res import ResMLP
# from model_c2 import Cnn1
from tqdm import tqdm, trange
import math 

import os
#CNN+MLP混合
flag=0
input_size_mag=5
input_size_mag=torch.tensor(input_size_mag)
output_size_mag=96
output_size_mag=torch.tensor(output_size_mag)
hidden_size_mag=512 #隐藏层的节点数量
hidden_size_mag=torch.tensor(hidden_size_mag)
num_layers_mag=5#隐藏层数量
num_layers_mag=torch.tensor(num_layers_mag)

input_size_position=96
input_size_position=torch.tensor(input_size_position)
output_size_position=5
output_size_position=torch.tensor(output_size_position)
hidden_size_position=1024 #隐藏层的节点数量
hidden_size_position=torch.tensor(hidden_size_position)
num_layers_position=5#隐藏层数量
num_layers_position=torch.tensor(num_layers_position)
##s
test_loss_select=0
flag=0
batch_size =80
learning_rate = 0.0001
epochs =500
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#device = torch.device('cpu')#('cuda:0')
print(device)
file_flag='RES_MUTUAL'
the = 0.1
count_filename = "model_Mutual_RES_count.txt"
Dir='Model_RES_MUTUAL/'
loss_folder='Model_RES_MUTUAL/loss_RES_MUTUAL/'
model_pkl_folder='Model_RES_MUTUAL/model_pkl_RES_MUTUAL/'
if not os.path.exists(Dir):  # 如果文件夹不存在
    os.makedirs(Dir)  # 创建文件夹
if not os.path.exists(loss_folder):  # 如果文件夹不存在
    os.makedirs(loss_folder)  # 创建文件夹

if not os.path.exists(model_pkl_folder):  # 如果文件夹不存在
    os.makedirs(model_pkl_folder)  # 创建文件夹
# 如果文件不存在，则初始化计数器为0
if not os.path.exists(count_filename):
    with open(count_filename, "w") as f:
        f.write("0")
# 读取计数器的当前值
with open(count_filename, "r") as f:
    count = int(f.read())
# 检查 modelss.txt 文件是否存在，如果不存在则创建
if not os.path.isfile('model_Mutual_RES_X_Y.txt'):
    open('model_Mutual_RES_X_Y.txt', 'w').close()
# 检查 modelss.txt 文件是否存在，如果不存在则创建
if not os.path.isfile('model_Mutual_RES_Y_X.txt'):
    open('model_Mutual_RES_Y_X.txt', 'w').close()
if not os.path.isfile('model_Mutual_RES_count.txt'):
    open('model_Mutual_RES_count.txt', 'w').close()
if count>=1:
    # 读取 model.txt 文件的最后一行，获取上次运行的 pkl 文件名
    with open('model_Mutual_RES_X_TO_Y.txt', 'r') as f:
        linesff = f.readlines()
        if linesff:
            last_lineff = linesff[-1].strip()
        else:
            last_lineff = ""
    # 读取 model.txt 文件的最后一行，获取上次运行的 pkl 文件名
    with open('model_Mutual_RES_Y_TO_X.txt', 'r') as f:
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
# x = np.loadtxt('d_new.txt', delimiter=',') # another list of numpy arrays (targets)
# y = np.loadtxt('l_new.txt', delimiter=',') # another list of numpy arrays (targets)
# x = np.loadtxt('d_test.txt', delimiter=',') # another list of numpy arrays (targets)
# y = np.loadtxt('l_test.txt', delimiter=',') # another list of numpy arrays (targets)
#senior = np.loadtxt('senior.txt',delimiter=',')
l1 = 101
l = np.size(x,0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

x = torch.tensor(x, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)  # float32

x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)  # float32
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
#senior = torch.tensor(senior,dtype=torch.float32).to(device)

train_dataset = data.TensorDataset(x_train, y_train)
test_dataset = data.TensorDataset(x_test, y_test)

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#shuffle是否对数据进行打乱
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
train = list(enumerate(train_dataloader))
test = list(enumerate(test_dataloader))

net_mag =  ResMLP(input_size_mag, output_size_mag, hidden_size_mag, num_layers_mag,device).to(device) #96->5
net_location = ResMLP(input_size_position, output_size_position, hidden_size_position, num_layers_position,device).to(device) #96->5
#if count>=1:
net_mag.load_state_dict(torch.load(last_lineff))
net_location.load_state_dict(torch.load(last_linecc))


optimizer_mag = optim.Adam(net_mag.parameters(),lr=learning_rate) ##根据实际的曲线去调整，看看那个过拟合,weight_decay=0.001
optimizer_location = optim.Adam(net_location.parameters(),lr=learning_rate)#,weight_decay=0.005
criteon_location = nn.MSELoss().to(device)

criteon = nn.SmoothL1Loss().to(device)

# x_new = x_new.T


g = 0
m = 0
losstest= np.array([])
TOTAL_LOSS_test =np.array([])
losstrain_location = np.array([])
losstest_x = np.array([])
losstrain_mag= np.array([])
epoch_train_total=np.array([])
epoch_train = np.array([])
epoch_test = np.array([])
epoch_test_total = np.array([])
epoch_train_x = np.array([])
TOTAL_LOSS_TRAIN = np.array([])
epoch_test_x = np.array([])
# predict_the = torch.tensor([200, 200, 200, 200, 200, 200, 200, 200, 200]).to(device)
predict_the = torch.from_numpy(np.zeros(384)).to(device)
predict_the = predict_the.view(1, -1)
predict_the = predict_the.repeat_interleave(batch_size, dim=0)

for epoch in range(epochs):
    for batch_idx in tqdm(range(len(train)), desc='train'):
        _, (data, target) = train[batch_idx]
        data, target = data.to(device), target.to(device)
        ones = torch.from_numpy(np.zeros(384)).to(device)
        total_euler_out = torch.from_numpy(np.zeros((data.size()[0], 96)))
        total_euler_out1 = torch.from_numpy(np.zeros((data.size()[0], 192)))
        fin_out1 = torch.from_numpy(np.zeros((data.size()[0], 5)))
        if data.size()[0] == predict_the.size()[0]:
            # x_input = torch.cat((data, target), dim=1).float()
            output = net_location(data,device).requires_grad_(True).to(device)
            predict_1 = net_mag(output,device).requires_grad_(True).to(device)
            # predict_y = predict_1.repeat(data.size()[0], 1).requires_grad_(True)
            loss_location = criteon_location(output,target)
            loss_mag = criteon(predict_1,data)
            loss = loss_location +the* loss_mag
            optimizer_mag.zero_grad()
            optimizer_location.zero_grad()
            #loss_location.backward(retain_graph=True)
            #loss_mag.backward(retain_graph=True)
            loss.backward(retain_graph=True)
            optimizer_mag.step()
            optimizer_location.step()
            #total_loss = criteon(fin_out,target)
            total_loss = criteon(output, target)
            losstrain_location= np.append(losstrain_location, total_loss.item())
            total_loss_x = criteon(predict_1, data)
            losstrain_mag = np.append(losstrain_mag, total_loss_x.item())
            TOTAL_LOSS_TRAIN=np.append(TOTAL_LOSS_TRAIN,  loss.item())


        # 最后的batch不足batch_size时
        else:
            output = net_location(data,device).requires_grad_(True).to(device)
            predict_1 = net_mag(output,device).requires_grad_(True).to(device)
            # predict_y = predict_1.repeat(data.size()[0], 1).requires_grad_(True)
            loss_location = criteon_location(output,target)
            loss_mag = criteon(predict_1, data)
            loss = loss_location +the* loss_mag
            optimizer_mag.zero_grad()
            optimizer_location.zero_grad()
            #loss_location.backward(retain_graph=True)
            #loss_mag.backward(retain_graph=True)
            loss.backward(retain_graph=True)
            optimizer_mag.step()
            optimizer_location.step()
            #total_loss = criteon(fin_out,target)
            total_loss = criteon(output, target)
            losstrain_location = np.append(losstrain_location, total_loss.item())
            total_loss_x = criteon(predict_1, data)
            losstrain_mag = np.append(losstrain_mag, total_loss_x.item())
            TOTAL_LOSS_TRAIN=np.append(TOTAL_LOSS_TRAIN,  loss.item())

    epoch_losstrain = np.mean(losstrain_location)
    epoch_losstrain_x = np.mean(losstrain_mag)
    epoch_totalloss_train=np.mean(TOTAL_LOSS_TRAIN)
    epoch_train_x = np.append(epoch_train_x, epoch_losstrain_x)
    epoch_train = np.append(epoch_train, epoch_losstrain)
    epoch_train_total=np.append(epoch_train_total,  epoch_totalloss_train)
    losstrain = np.array([])
    losstrain_x = np.array([])

    print('Train set :epoch{} 磁场Averge loss: {:.4f}****位置误差: {:.4f}*****Total loss:{:.4f}\n'.format(
            epoch,epoch_losstrain_x,epoch_losstrain,epoch_totalloss_train
        ))

    test_loss= 0
    total_loss_x  = 0

    for batch_idx in tqdm(range(len(test)), desc='test'):
        _, (test_data, test_target) = test[batch_idx]
        test_data, test_target = test_data.to(device), test_target.to(device)

        if test_data.size()[0] == predict_the.size()[0]:


            output = net_location(test_data,device).requires_grad_(True).to(device)
            predict_1 = net_mag(output,device).requires_grad_(True).to(device)
            # predict_y = predict_1.repeat(,devicedata.size()[0], 1).requires_grad_(True)

            test_loss = criteon_location(output, test_target)
            losstest = np.append(losstrain, test_loss.item())
            total_loss_x = criteon(predict_1, test_data)
            loss_zhengzehua =the* total_loss_x+test_loss 
            losstest_x = np.append(losstrain_x, total_loss_x.item())
            TOTAL_LOSS_test=np.append(TOTAL_LOSS_test, loss_zhengzehua .item())

        else:
            output = net_location(test_data,device).requires_grad_(True).to(device)
            predict_1 = net_mag(output,device).requires_grad_(True).to(device)
            # predict_y = predict_1.repeat(data.size()[0], 1).requires_grad_(True)

            test_loss = criteon_location(output, test_target)
            losstest = np.append(losstrain, test_loss.item())
            total_loss_x = criteon(predict_1, test_data)
            loss_zhengzehua =the* total_loss_x+test_loss 
            losstest_x = np.append(losstrain_x, total_loss_x.item())
            TOTAL_LOSS_test=np.append(TOTAL_LOSS_test , loss_zhengzehua.item())
    epoch_losstest = np.mean(losstest)
    epoch_losstest_x = np.mean(losstest_x)
    epoch_totalloss_test=np.mean(TOTAL_LOSS_test)
    epoch_test = np.append(epoch_test, epoch_losstest)
    epoch_test_x = np.append(epoch_test_x, epoch_losstest_x)
    epoch_test_total=np.append(epoch_test_total,epoch_totalloss_test)
    temp=epoch_losstest
    # 读取计数器的当前值
    if(test_loss_select>temp):
            with open(count_filename, "r") as f:
                count = int(f.read())

            losstest = np.array([])
            losstest_x = np.array([])
            g = epoch_losstest
            print('当前最优Test Epoch: {}磁场Averge loss: {:.4f}****位置误差: {:.4f}*****Total loss:{:.4f}\n'.format(
                   epoch, epoch_losstest_x,epoch_losstest,epoch_totalloss_test
                ))
            torch.save(net_mag.state_dict(), model_pkl_folder+'MUTUAL_X_Y_' + str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count) +'_' + str(the) + '.pkl')
            torch.save(net_location.state_dict(), model_pkl_folder+'MUTUAL_Y_X_' + str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count) +'_' + str(the) +  '.pkl')
            test_loss_select=temp
    else:
        print('非最优解Test set : 磁场Averge loss: {:.4f}****位置误差: {:.4f}*****Total loss:{:.4f}\n'.format(
                    epoch_losstest_x,epoch_losstest,epoch_totalloss_test
                ))
    if(flag==0):
        test_loss_select=epoch_losstest
        flag=1  
        
        
#     if g < 10:
#         torch.save(net_mag.state_dict(), 'model_pkl/finlf'+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count)+'.pkl')
#         # 在程序运行结束后，将本次运行的 pkl 文件名追加到 model.txt 文件的末尾
#         with open('modelff.txt', 'a') as f:
#             f.write('model_pkl/finlf'+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count)+'.pkl\n')  # 将 pkl 文件名替换为你的实际文件名
#         torch.save(net_location.state_dict(), 'model_pkl/finlc'+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count)+'.pkl')
#         with open('modelcc.txt', 'a') as f:
#             f.write('model_pkl/finlc' + str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count) + '.pkl\n')
#         print("找到较优解:", g)
#         m = 1
# if m == 1:
#     print("已经找到较优解")
# else:
#     print("没有找到较好的解")

    # 在程序运行结束后，将本次运行的 pkl 文件名追加到 model.txt 文件的末尾
with open('model_Mutual_RES_X_TO_Y.txt', 'a') as f:
        f.write(model_pkl_folder+'MUTUAL_X_Y_' + str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count) +'_' + str(the) + '.pkl\n')  # 将 pkl 文件名替换为你的实际文件名

with open('model_Mutual_RES_Y_TO_X.txt', 'a') as f:
        f.write(model_pkl_folder+'MUTUAL_Y_X_' + str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count) +'_' + str(the) +  '.pkl\n')

import matplotlib.pyplot as plt
#
with open(loss_folder+'磁场误差train'+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count) +'_' + str(the) + '.txt', 'a+') as f:
    np.savetxt(f,epoch_train_x)#np.savetxt(r'test.txt', x, fmt='%d', newline='-|-')
    f.close()
with open(loss_folder+'磁场误差test'+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count)+'_' + str(the) + '.txt', 'a+') as f:
    np.savetxt(f,epoch_test_x)#np.savetxt(r'test.txt', x, fmt='%d', newline='-|-')
    f.close()
with open(loss_folder+'位置误差train'+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count)+'_' + str(the) + '.txt', 'a+') as f:
    np.savetxt(f,epoch_train)#np.savetxt(r'test.txt', x, fmt='%d', newline='-|-')
    f.close()
with open(loss_folder+'位置误差test'+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count)+'_' + str(the) + '.txt', 'a+') as f:
    np.savetxt(f,epoch_test)#np.savetxt(r'test.txt', x, fmt='%d', newline='-|-')
    f.close()
with open(loss_folder+'TOTAL_LOSS_TRAIN'+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count)+'_' + str(the) + '.txt', 'a+') as f:
    np.savetxt(f,epoch_train_total)#np.savetxt(r'test.txt', x, fmt='%d', newline='-|-')
    f.close()
with open(loss_folder+'TOTAL_LOSS_TEST'+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count)+'_' + str(the) + '.txt', 'a+') as f:
    np.savetxt(f,epoch_test_total)#np.savetxt(r'test.txt', x, fmt='%d', newline='-|-')
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
plt.title('Location LOSS')
plt.plot(plot_epochs,epoch_train, label='Training Loss',color='b')
plt.plot(plot_epochs, epoch_test,label='Test Loss',color='r')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(loss_folder+'Mutual_RES_location'+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count)+'the='+str(the)+'.jpg',dpi=200)
plt.close()

#绘制训练集和测试集loss下降曲线
plot_epochs = []
for i in range(epochs):
    plot_epochs.append(int(i))
plot_epochs = np.array(plot_epochs)
plt.title('MAG LOSS')
plt.plot(plot_epochs,epoch_train_x, label='Training Loss',color='b')
plt.plot(plot_epochs, epoch_test_x,label='Test Loss',color='r')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(loss_folder+'Mutual_RES_Mag'+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count)+'the='+str(the)+'.jpg',dpi=200)
# 将计数器的值加1，并保存到文件中
plt.close()

#绘制训练集和测试集loss下降曲线
plot_epochs = []
for i in range(epochs):
    plot_epochs.append(int(i))
plot_epochs = np.array(plot_epochs)
plt.title('TOTAL LOSS')
plt.plot(plot_epochs,epoch_train_total, label='Training Loss',color='b')
plt.plot(plot_epochs, epoch_test_total,label='Test Loss',color='r')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(loss_folder+'Mutual_RES_Total'+str(learning_rate)+'_' + str(batch_size)+'_' + str(epochs)+'_' + str(count)+'the='+str(the)+'.jpg',dpi=200)
# 将计数器的值加1，并保存到文件中
count += 1
with open(count_filename, "w") as f:
    f.write(str(count))