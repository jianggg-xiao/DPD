import torch
import torch.nn as nn
import scipy.io as scio
import time
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class RealConv(nn.Module):
    def __init__(self,inchnnl, outchnnl, k):
        super(RealConv, self).__init__()
        self.conv = nn.Conv2d(inchnnl, outchnnl, (k, 2), padding='same', bias=False)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = torch.cat((x.real, x.imag), dim=-1)
        y = self.conv(x)
        y = y[:, :, :, 0] + y[:, :, :, 1] * 1j
        return y


class RBF(nn.Module):
    def __init__(self, num_centers, num_branches):
        super(RBF, self).__init__()
        self.num_centers = num_centers
        self.c = nn.Parameter(torch.zeros(num_branches, 1, self.num_centers, dtype=torch.cfloat))
        self.sigma = nn.Parameter(torch.zeros(num_branches, 1, self.num_centers, dtype=torch.cfloat))
        self.fc = nn.Parameter(torch.zeros(num_branches, self.num_centers, 1, dtype=torch.cfloat))

    def forward(self, x):
        x_abs = x.abs().unsqueeze(-1).repeat(1, 1, self.num_centers)
        distr = torch.exp(-(x_abs - self.c.real) ** 2 * self.sigma.real)
        disti = torch.exp(-(x_abs - self.c.imag) ** 2 * self.sigma.imag)
        outr = torch.matmul(distr, self.fc.real).squeeze(-1)
        outi = torch.matmul(disti, self.fc.imag).squeeze(-1)
        out = outr + outi * 1j
        return out * x + x


class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        xr, xi = x.real, x.imag
        yr = torch.relu(xr)
        yi = torch.relu(xi)
        return yr + yi * 1j


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        xr, xi = x.real, x.imag
        yr = torch.sigmoid(xr)
        yi = torch.sigmoid(xi)
        return yr + yi * 1j


class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        xr, xi = x.real, x.imag
        yr = torch.tanh(xr)
        yi = torch.tanh(xi)
        return yr + yi * 1j


class MyModel(nn.Module):
    def __init__(self, num):
        super(MyModel, self).__init__()
        # self.L1 = RealConv(1, num, 2 * num + 1)
        # torch.nn.init.constant_(self.L1.conv.weight, 0)
        self.L1 = nn.Conv1d(1, num, 2 * num + 1, padding='same', dtype=torch.cfloat, bias=False)
        torch.nn.init.constant_(self.L1.weight, 0 + 0j)
        self.L2 = RBF(num, num)
        # self.L3 = RealConv(num, 1, 1)
        # torch.nn.init.constant_(self.L3.conv.weight, 1e-6)
        self.L3 = nn.Conv1d(num, 1, 1, padding='same', dtype=torch.cfloat, bias=False)
        torch.nn.init.constant_(self.L3.weight, 1e-6 + 0j)

    def forward(self, x):
        x0 = x.cfloat().unsqueeze(0).unsqueeze(0)
        x1 = self.L1(x0).squeeze(0)
        x2 = self.L2(x1)
        x3 = self.L3(x2.unsqueeze(0)).squeeze(0)
        return x3.squeeze(0)


def run_single_iter(dl, model, optimizer=None, scheduler=None):
    for xb, yb in dl:
        yout = model(xb.to(device))
        err = yb.to(device) - yout
        if optimizer:
            loss = (err.abs() ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if scheduler:
        scheduler.step()
        # print('train_time:', round(time.time() - start,2))


def model_eval(model, x, y):
    x, y = x.to(device), y.to(device)
    yout = model(x) * ymax
    e = yout - y
    NMSE = 10 * torch.log10((e.abs() ** 2).mean() / ((y.abs() ** 2).mean()))
    return NMSE.item(), yout


if __name__ == "__main__":
    # 数据预处理
    file_addr = '../Data/'
    file_name = 'Simulation_MPDPD_Data'
    mat = scio.loadmat(file_addr + file_name)
    x = torch.from_numpy(mat['x'])
    y = torch.from_numpy(mat['d'])
    L1 = int(y.shape[1] * 0.5)
    L2 = int(y.shape[1] * 0.6)
    x_tra, y_tra = x[0, :L1], y[0, :L1]
    x_tra_all = []
    y_tra_all = []
    block_size = 1024
    for i in range(1000):
        start = np.random.randint(len(x_tra) - block_size)
        x_tra_all.append(x_tra[start:start + block_size])
        y_tra_all.append(y_tra[start:start + block_size])
    x_tra_all = np.hstack(x_tra_all)
    y_tra_all = np.hstack(y_tra_all)
    xmax = x.abs().max()
    ymax = y.abs().max()
    tra_dl = DataLoader(TensorDataset(x_tra_all / xmax, y_tra_all / ymax), batch_size=1024)

    dict = {}
    for num_neuron in range(6, 7):
        model = MyModel(num_neuron).to(device)
        param_num = sum([param.nelement() for param in model.parameters()]) * 2
        # if num_neuron > 1:
        #     model_tmp = torch.load('../Models/' + file_name + '_GNAS-CVCNN_RConv_' + str(num_neuron - 1) + '.pth')
        #     model.L1.weight.data[:num_neuron - 1, :, 1:-1] = model_tmp.L1.weight.data
        #     model.L2.c.data[:num_neuron - 1, :, :-1] = model_tmp.L2.c.data
        #     model.L2.sigma.data[:num_neuron - 1, :, :-1] = model_tmp.L2.sigma.data
        #     model.L2.fc.data[:num_neuron - 1, :-1, :] = model_tmp.L2.fc.data
        #     model.L3.weight.data[:, :num_neuron - 1, :] = model_tmp.L3.weight.data
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        NMSE_all = []
        for epoch in range(400):
            '''训练模型'''
            run_single_iter(tra_dl, model, optimizer, scheduler)
            '''模型验证'''
            NMSE_tra = round(model_eval(model, x[0, :L1] / xmax, y[0, :L1])[0], 2)
            NMSE_val = round(model_eval(model, x[0, L1:L2] / xmax, y[0, L1:L2])[0], 2)
            print(epoch, '/', num_neuron, 'train:', NMSE_tra, 'valid:', NMSE_val, 'param_num:', param_num)
            NMSE_all.append([NMSE_tra, NMSE_val])
        dict['out'] = model_eval(model, x[0, L2:] / xmax, y[0, L2:])[1].cpu().detach().numpy()
        dict[str(num_neuron) + '_NMSE_all'] = np.array(NMSE_all)
        dict[str(num_neuron) + '_param_num'] = param_num
        scio.savemat('../DataSave/' + file_name + '_GNAS-CVCNN_RConv_' + str(num_neuron) + '.mat', dict)
        torch.save(model, '../Models/' + file_name + '_GNAS-CVCNN_RConv_' + str(num_neuron) + '.pth')