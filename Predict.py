import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from PIL import Image
import torchvision
from tqdm import tqdm


class Config:
    def __init__(self):
        self.cpu_num_threads = 12
        self.workdir = "./"
        self.data_folder = self.workdir + "data/" + "available_data/"
        self.result_folder = "./result/"
        self.model_path = self.result_folder + "Resnet.pth"
        self.folder_classes = ["train", "test", "ind"]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 32
        self.seed = 2022
        torch.manual_seed(self.seed)
        os.chdir(self.workdir)
        torch.set_num_threads(self.cpu_num_threads)
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)


class MyDataset(Dataset):
    def __init__(self, imgs_path, labels_num):
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((837, 837)),
            torchvision.transforms.ToTensor()
        ])
        self.imgs_path = imgs_path
        self.labels = labels_num

    def __getitem__(self, index):
        img_path, label = self.imgs_path[index], self.labels[index]
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil)
        if len(img_np.shape) == 2:
            img_np = np.repeat(img_np[:, :, np.newaxis], 3, axis=2)
            img_pil = Image.fromarray(img_np)
        img = self.transform(img_pil)
        label = torch.tensor(label, dtype=torch.int64)
        return img.type(torch.float32), label

    def __len__(self):
        return len(self.labels)


class Fc(torch.nn.Module):
    def __init__(self, in_features):
        super(Fc, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=in_features, out_features=256)
        self.linear2 = torch.nn.Linear(in_features=256, out_features=64)
        self.linear3 = torch.nn.Linear(in_features=64, out_features=5)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        x = F.dropout(F.relu(self.linear1(input)), p=0.5)
        x = F.dropout(F.relu(self.linear2(x)), p=0.5)
        logits = self.linear3(x)
        return logits


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        resnet_model = torchvision.models.resnet152(pretrained=True)
        in_features = resnet_model.fc.in_features
        resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
        self.resnet_model = resnet_model
        self.fc = Fc(in_features)

    def forward(self, input):
        x = self.resnet_model(input)
        logits = self.fc(x)
        return logits


class Predict(Config):
    """
        计算文件夹(target_floder)内所有图片的预测准确率；
    """

    def __init__(self, target_floder):
        super(Predict, self).__init__()
        self.target_floder = target_floder

    def get_imgs_path_and_labels(self, target_floder=None):
        # 由图片名称判断图片类别，并以列表形式返回
        if target_floder is None:
            target_floder = self.target_floder
        imgs_path, labels = [], []
        imgs_path = glob.glob(target_floder + "*.jpg")
        for img_path in imgs_path:
            labels.append(int(img_path[-5]))
        return imgs_path, labels

    def _test(self, test_dl, model):
        model.eval()
        test_data_num = len(test_dl.dataset)
        acc = 0
        with torch.no_grad():
            for x, y in tqdm(test_dl):
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x)
                acc += (pred.argmax(1) == y).sum().item()
            acc = acc / test_data_num
            return acc

    def get_accuracy(self, test_ds=None):
        if test_ds is None:
            imgs_path, labels = self.get_imgs_path_and_labels()
            test_ds = MyDataset(imgs_path, labels)
        test_dl = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        model = Model().to(self.device)
        model.load_state_dict(torch.load(self.model_path))
        acc = self._test(test_dl, model)
        return acc


if __name__ == '__main__':
    test_data_floder = "./data/available_data/test/"
    ind_data_floder = "./data/available_data/ind/"
    test_predict = Predict(target_floder=test_data_floder)
    test_acc = test_predict.get_accuracy()
    ind_predict = Predict(target_floder=ind_data_floder)
    ind_acc = ind_predict.get_accuracy()
    print("测试集的准确率为：{}， 独立验证集的准确率为：{}".format(round(test_acc, 4), round(ind_acc, 4)))
    """
    输出结果：
        测试集的准确率为：0.9061， 独立验证集的准确率为：0.9023
    """
