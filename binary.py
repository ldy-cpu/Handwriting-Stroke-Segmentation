import os
from PIL import Image
import numpy as np
import configg

path = "./data/{}/all/test/label_result".format(configg.name)
f = os.listdir(path)
for k in f:
    img_path = path + "/" + k
    ff = os.listdir(img_path)
    for kk in ff:
        single_path = img_path + "/" + kk
        save_path = img_path + "/" + "bin_" + kk
        img = Image.open(single_path)
        img = img.convert("L")
        img = np.array(img)

        img = img > 128

        img = Image.fromarray(img)
        img.save(save_path)
