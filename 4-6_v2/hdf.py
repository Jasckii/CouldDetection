import h5py
import os
from PIL import Image
import numpy as np
hd5_path = r'D:\bing下载\MOD35_L2.A2025001.0000.061.2025001132506.hdf'  # 读取路径
sotre_path = r'F:\桌面\jpg'  # 存储路径
hr_dataset = h5py.File(hd5_path)

for i in range(len(hr_dataset['image'])):
    y = hr_dataset['image'][i]
    if not os.path.exists(sotre_path):
        os.mkdir(sotre_path)
    j = 0
    for j in range(len(y[0])):
        x = y[0][j,:,:]
        x = np.reshape(x, (512, 512))
        # name = os.path.join(sotre_path, str(k) + '_' + str(j) + '.png')
        for j in range(len(y[0])):  # range(4)
            name = os.path.join(sotre_path, str(i) + '_' + str(j) + '.jpg')
            img1 = Image.fromarray(x)
            img1.convert('RGB').save(name)
