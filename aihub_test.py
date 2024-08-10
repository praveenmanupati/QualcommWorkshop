import numpy as np
from PIL import Image
import qai_hub as hub 
from typing import Dict, List


# Convert the image to numpy array of shape [1, 3, 224, 224]
image = Image.open("kitten.jpg").resize((224, 224))
img_array = np.array(image, dtype=np.float32)

# Ensure correct layout (NCHW) and re-scale
input_array = np.expand_dims(np.transpose(img_array / 255.0, (2, 0, 1)), axis=0)

# Step 1: Create ONNX Runtime Compiling Job 
device_name = hub.Device("Snapdragon X Elite CRD")
                        
#Compiling ONNX runtime 
compile_job = hub.submit_compile_job(
                name="MobileNet",
                model="mobilenet_v2.onnx",
                input_specs=dict(img_input=(1,3, 224, 224)),
                #Select target runtime & compute unit as cpu or npu as required, default is npu.
                options="--target_runtime onnx --compute_unit npu",
                device=device_name, 
                )

# Run inference using the on-device model on the input image
inference_job = hub.submit_inference_job(
    model=compile_job.get_target_model(),
    device=device_name,
    inputs=dict(img_input=[input_array]),
)
assert isinstance(inference_job, hub.InferenceJob)

# Get the on-device output
on_device_output: Dict[str, List[np.ndarray]] = inference_job.download_output_data()  # type: ignore

# Calculate probabilities for the on-device output
output_name = list(on_device_output.keys())[0]
out = on_device_output[output_name][0]
on_device_probabilities = np.exp(out) / np.sum(np.exp(out), axis=1)

# Read the class labels for imagenet
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]


# Print top five predictions for the on-device model
print("Top-5 On-Device predictions:")
top5_classes = np.argsort(on_device_probabilities[0], axis=0)[-5:]
for c in reversed(top5_classes):
    print(f"{c} {categories[c]:20s} {on_device_probabilities[0][c]:>6.1%}")