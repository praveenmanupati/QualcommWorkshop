#preprocess_image.py
                        
import numpy as np
#Qualcomm utility for pre-/postprocessing of input/outputs in model inference
import qc_utils 
                        
img_path = r"C:\Mobilenet\kitten.jpg" 
raw_img = qc_utils.preprocess(img_path, True)
raw_img = raw_img.astype(np.float32) 
raw_save_path = img_path.replace('jpg','raw')
raw_img.tofile(raw_save_path)
                        
#create input list
with open(r"C:\Mobilenet\input_list.txt", "w") as file: 
    file.write(raw_save_path) 