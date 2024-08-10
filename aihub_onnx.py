# File_name : aihub_onnx.py 
                        
import qai_hub as hub 
import numpy as np
#Qualcomm utility for pre-/postprocessing of input/outputs in Mobilenet model inference 
import qc_utils
                        
# Step 1: Create ONNX Runtime Compiling Job 
device_name =hub.Device("Snapdragon X Elite CRD")
                        
#Compiling ONNX runtime 
compile_job = hub.submit_compile_job(
                name="MobileNet",
                model="mobilenet_v2.onnx",
                input_specs=dict(img_input=(1,3, 224, 224)),
                #Select target runtime & compute unit as cpu or npu as required, default is npu.
                options="--target_runtime onnx --compute_unit npu",
                device=device_name, 
                )
                        
# Step 2: Input/Output handling, Generate raw input 
img_path = r"C:\MobileNet\kitten.jpg" 
raw_img = qc_utils.preprocess(img_path) 
                        
# Step 3: Run Model inferencing Job 
                        
inference_job = hub.submit_inference_job( 
                  name="MobileNet",
                  model=compile_job.get_target_model(),
                  device=device_name, 
                  inputs={"img_input": [raw_img]},
                  options="--compute_unit npu"
                  ) 
                        
# Get on-device output 
on_device_output = inference_job.download_output_data() 
prediction = on_device_output[list(on_device_output.keys())[0]] 
                        
# Step 4: Output postprocessing 
qc_utils.postprocess(prediction)