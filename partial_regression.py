from PIL import Image
import numpy as np



p =  "./data/创/all/test_170epoch_6class_reverse/image/"
pic = "004210-06-05-1.png"
ori_path = p + pic

s = "./data/创/all/test_170epoch_6class_reverse/label_result/"
stroke_name = "5.png"
stroke_path = s + pic + "/" + stroke_name

ori = Image.open(ori_path)
stroke = Image.open(stroke_path)

ori = np.array(ori)
stroke = np.array(stroke)




