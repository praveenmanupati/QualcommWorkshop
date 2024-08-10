import numpy as np
#Qualcomm utility for pre-/postprocessing of input/outputs in model inference

import qc_utils

#cpu_output_path = r"C:\Mobilenet\output_cpu\Result_0\_590.raw" 
htp_output_path = r"C:\MobileNet\output_htp\Result_0\class_logits.raw"

#print("cpu: Result ") 
#qc_utils.postprocess(cpu_output_path) 
print("\nhtp: Result ")

qc_utils.postprocess(htp_output_path)