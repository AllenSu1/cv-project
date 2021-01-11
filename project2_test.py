import csv
import cv2
import os
import time
from PIL import Image
from model import *
import torch
import numpy as np
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda")

# 读入图片
image_path = r'C:\Users\Allen\Desktop\finalproject_img\test'
classes = os.listdir(r'C:\Users\Allen\Desktop\finalproject_img\train')

test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


# model
model = torchvision.models.resnet101(pretrained=True, progress=True)
# 提取參數fc的輸入參數
fc_features = model.fc.in_features
# 將最後輸出類別改為20
model.fc = nn.Linear(fc_features, 2)
# 輸入訓練好權重
model.load_state_dict(torch.load("model/1.pth"))

# # 遷移學習 -> frezee
# for name, parameter in model.named_parameters():
#     # print(name)
#     if name == 'layer4.0.conv1.weight':
#         break
#     # if name == 'fc.weight':
#     #     break
#     parameter.requires_grad = False

model.to(device)
model.eval()

total_time = 0
result = []
good = 0
bad = 0
# for i in tqdm(range(1, 26)):
for i in range(1, 26):    
    start = 0
    end =0
    com_img = os.path.join(image_path, '%s.jpg' % i)
    test_data = Image.open(com_img).convert('RGB')
    data_transforms = test_transforms(test_data).to(device)
    start = time.time()
    pred = model(data_transforms[None, ...])
    # print(data_transforms[None, ...].shape)
    predict_y = torch.max(pred, dim=1)[1]
    result.append(classes[int(predict_y)])
    if classes[int(predict_y)] == 'bad':
        bad+=1
    else:
        good+=1

    end = time.time()
    print('img%2d:      執行時間 : %.4f     classes：%4s'  %(i,(end - start),classes[int(predict_y)]))
    total_time += (end - start)
    
    # print(result)
sample = pd.DataFrame({
    'image': pd.read_csv(r'D:\GitHub\AllenSu1\ComputerVision_FinalProject\result.csv').image,
    'result': result[:]
})
print('----------average----------')
print('判斷成good有%d個、bad有%d個' %(good,bad))
acc = ((bad/25)*100)
print('準確率：{:.2f}%'.format(acc))
print('平均執行時間: %.4f' %(total_time/7))
sample.to_csv('result1.csv', index=False)
