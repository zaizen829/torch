import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline

from torchvision import datasets

device = torch.device("cpu")
use_pretrained = True  # 学習済みのパラメータを使用
model = models.densenet161(pretrained=use_pretrained).to(device)

model.classifier = nn.Linear(in_features=2208, out_features=6, bias=True).to(device)

#モデルを読み込む
device = torch.device("cuda")
#model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load("checkpoint.pt"))
model.to(device)

transform_test = transforms.Compose([
    transforms.Resize(224), #サイズ合わせ
    transforms.CenterCrop(224),
    transforms.ToTensor() ,              #型変換(?)
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  #https://teratail.com/questions/234027

    ])


your_datasets = datasets.ImageFolder(root="jikken_hanbetu", transform=transform_test)


classes = ("jui","ocha","piza","pop","pre","toppo")


# 評価モードにする
model = model.eval()



#　※ディレクトリの順番がa～z順になっていない場合は多分ずれるのでif分の中を書き換える
jui = 0
ocha = 0
piza = 0
pop =0
pre = 0
toppo = 0
totall= 0

for ii in range(600):
   TEST = your_datasets[ii][0].unsqueeze(0)
   TEST = TEST.cuda()
   y_predict = model.forward(TEST)
   print(ii,  " : ", classes[y_predict[0].argmax()])

   if y_predict[0].argmax() == your_datasets[ii][1]:
      if 0 <= ii <=99:
           jui+=1
           totall+=1
      if 100 <= ii <=199:
           ocha+=1
           totall+=1
      if 200 <= ii <=299:
           piza+=1
           totall+=1
      if 300 <= ii <=399:
           pop+=1
           totall+=1
      if 400 <= ii <=499:
           pre+=1
           totall+=1
      if 500 <= ii <=599:
           toppo+=1
           totall+=1


print("正解数")
print("jui  ：",jui)
print("ocha      ：",ocha)
print("piza       ：",piza)
print("pop       ：",pop)
print("pre      ：",pre)
print("toppo       ：",toppo)

print("------------------")
print("合計正解数：",totall)
print("正答率    ：",(totall/600)*100,"%")
