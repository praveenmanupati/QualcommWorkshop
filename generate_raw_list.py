#generate_raw_list.py 

import numpy as np 
import os
#Qualcomm utility for pre-/postprocessing of input/outputs in model inference
import qc_utils 

#path of directory containing jpg images sampled from the validation dataset.
image_dir_path = r"C:\MobileNet\quant_jpg" 

files=os.listdir(image_dir_path)

#path of directory containing raw files 
raw_dir_path = r"C:\MobileNet\quant_raw"
if not os.path.exists(raw_dir_path):
    os.mkdir(raw_dir_path)

input_path_list =[]

#convert jpg to raw
for file_name in files:
    file_path = os.path.join(image_dir_path, file_name)
    raw_img = qc_utils.preprocess(file_path,True)
    save_path = os.path.join(raw_dir_path, file_name.split('.')[0]+'.raw')
    raw_img.tofile(save_path)
    input_path_list.append(save_path)

#create raw list
with open(r"C:\MobileNet\raw_list.txt", "w") as f:
    for path in input_path_list:
        f.write(path)
        f.write('\n')