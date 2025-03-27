import torch.nn as nn
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from net.cnn import CNN5


def win_function(m=20):
    return np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (m - 1)) for n in range(m)])


def add_window(image):
    window = win_function(m=np.size(image, 0)) * win_function(m=np.size(image, 1)).reshape(-1, 1)
    return image * window


def add_filter(img):
    return cv2.GaussianBlur(img, (5, 5), 1)


def soft_max(data):
    m = nn.Softmax(dim=1)
    result = m(data)
    return result[0].cpu().numpy().tolist()


def fft(img):
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)

    s = np.log(np.abs(f_shift) / f_shift.size ** 0.5 + 1)
    s_new = (s - np.min(s)) / (np.max(s) - np.min(s))

    t_s = torch.tensor(s_new, dtype=torch.float32)
    img_fft = t_s.unsqueeze(0)
    return img_fft


class Prediction:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img_size = 224
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.norm = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size)])
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        self.classes_name = ['p1', 'pm', 'pg', 'cm',
                             'p2', 'pmm', 'pgg', 'pmg,cmm',
                             'p4', 'p4m,p4g',
                             'p3', 'p3m1,p31m',
                             'p6', 'p6m']
        self.label_matrix = [[1, 0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 1, 0],
                             [1, 0, 0, 0, 0, 0, 1],
                             [1, 0, 0, 0, 0, 1, 1],

                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 1, 0],
                             [0, 1, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 1, 1],

                             [0, 1, 1, 0, 0, 0, 0],
                             [0, 1, 1, 0, 0, 1, 0],

                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0, 1, 0],

                             [0, 1, 0, 1, 1, 0, 0],
                             [0, 1, 0, 1, 1, 1, 0]]
        pth_path = 'net/'
        self.pth_files = [pth_path + '1-epoch_18_20_2025-03-14_22.47.10.pkl',
                          pth_path + '2-epoch_8_20_2025-03-15_00.01.15.pkl',
                          pth_path + '3-epoch_15_20_2025-03-15_01.14.48.pkl',
                          pth_path + '4-epoch_16_20_2025-03-15_02.20.30.pkl',
                          pth_path + '5-epoch_13_20_2025-03-15_03.35.12.pkl',
                          pth_path + '6-epoch_11_20_2025-03-16_14.12.00.pkl'
                          ]
        self.prediction()

    def processing(self, image):
        image = np.array(self.norm(image))
        img = 255 * ((image - np.min(image)) / (np.max(image) - np.min(image)))
        window_img = add_window(img)
        filter_img = add_filter(window_img)
        fft_img = fft(filter_img)
        norm_img = self.normalize(fft_img)
        return norm_img.unsqueeze(dim=0)

    def error_distance(self, differ):
        distance = []
        for label in self.label_matrix:
            count = 1 - differ[5 + label[5] * 2 + label[6]]
            for i in range(5):
                count += abs(label[i] - differ[i])
            distance.append(count)
        return distance

    def prediction(self):
        image = Image.open(self.img_path).convert('L')
        img = self.processing(image)
        img = img.to(self.device)

        pre = []
        for i, pth_file in enumerate(self.pth_files):
            pre_trained_net = CNN5(self.img_size, 4 if i == 5 else 2)
            pre_trained_net.load_state_dict(torch.load(pth_file))
            model = pre_trained_net.to(self.device)
            model.eval()
            with torch.no_grad():
                outputs = model(img)
                pre.extend(soft_max(outputs)) if i == 5 else pre.append(soft_max(outputs)[1])
        distance = self.error_distance(pre)
        result = [self.classes_name[index] for (index, value) in enumerate(distance) if value == min(distance)]
        print('prediction:', result)

if __name__ == '__main__':
    Prediction(img_path='Example-p6m.png')
