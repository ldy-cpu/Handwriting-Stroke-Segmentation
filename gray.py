import os
from  PIL import Image
import numpy as np

path = r'./data/创/4/val/image'
f = os.listdir(path)
img_path = path + '/'
for k in f:
    img = Image.open(img_path + k)
    img = img.convert('L')
    img = np.array(img)


    # img = sauvola(img, k=0.1, kernerl=(31, 31))
    # img = img > 110

    img = Image.fromarray(img)
    img.save(img_path +  k)