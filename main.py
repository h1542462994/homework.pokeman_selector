import os
import torch
import torch.nn as nn
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

model_path = './model/model.pth'

data_transform = transforms.Compose(
    [
        transforms.Resize(480),
        transforms.CenterCrop(480),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ]
)


class FileLabels:
    def __init__(self, path):
        # self.path = path
        li = []
        dir_s = os.listdir(path)
        for _index in range(0, len(dir_s)):
            dir_path = os.path.join(path, dir_s[_index])
            for _file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, _file)
                li.append([_index, file_path])
        self.li = li

    def __len__(self):
        return len(self.li)

    def __getitem__(self, index):
        return self.li[index]


class MDataSet(Dataset):
    def __init__(self, path, _train=True, transform=None):
        self.path = path
        self.li = FileLabels(path)
        self.train = _train
        if transform is None:
            self.transform = data_transform
        else:
            self.transform = transform

    def __getitem__(self, index):
        item = self.li[index]
        label = item[0]
        img = Image.open(item[1])
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.li)


train_dataset = MDataSet('./dataset/train', transform=data_transform)
test_dataset = MDataSet('./dataset/test', transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True, num_workers=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # batch*3*480*480

        self.con1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),  # 激活函数
        )

        # batch*16*240*240

        self.con2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )

        # batch*32*120*120

        self.con3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )

        # batch*64*60*60

        self.fc = nn.Sequential(
            nn.Linear(64 * 60 * 60, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 12)
        )

        # 使用Adam优化算法
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)
        self.los = torch.nn.CrossEntropyLoss()

    def forward(self, _input):
        out = self.con1(_input)
        out = self.con2(out)
        out = self.con3(out)
        out = out.view(-1, 64 * 60 * 60)
        out = self.fc(out)
        return out

    def train_model(self, x, y):
        out = self.forward(x)
        loss = self.los(out, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        print('loss:', loss.item())

    def test_model(self, x):
        return self.forward(x)


def train():
    net = Net().cuda()
    ticks = 20
    for tick in range(ticks):
        print('Epoch {}/{}:'.format(tick, ticks - 1))
        for i, (data, y) in enumerate(train_loader):
            net.train_model(data.cuda(), y.cuda())
        _test('train', net, train_loader)
        net.eval()
        _test('test', net, test_loader)
        torch.save(net.state_dict(), model_path)
        print()


def test():
    t_net = Net().cuda()
    t_net.eval()
    t_net.load_state_dict(torch.load(model_path))

    print('final_test:')
    _test('test', t_net, test_loader)


def _test(name, net:Net, data: DataLoader):
    test_correct = 0
    test_total = 0
    for image_label in data:
        images, labels = image_label
        torch.no_grad()
        outputs = net.test_model(images.cuda())
        mm, prediction = torch.max(outputs.data, 1)
        test_correct += int(torch.sum(prediction.data.cuda() == labels.data.cuda()))
        test_total += len(labels.data)
    print('{}_acc: {}'.format(name, test_correct / test_total))


if __name__ == '__main__':
    train()
    test()
