import os                   
import numpy as np
#Qualcomm utility for pre-/postprocessing of input/outputs in model inference
import qc_utils 
import cv2

INPUT_LIST_PATH = r"C:\Mobilenet\input_list.txt"
OUTPUT_DIR = r"C:\Mobilenet\output_htp"
FRAMES_DIR = r"C:\MobileNet\Frames"

def pre_process(img_path):
    #img_path = r"C:\Mobilenet\kitten.jpg" 
    print(f"Preprocessing the image: {img_path}")
    raw_img = qc_utils.preprocess(img_path, True)
    print(raw_img.shape)
    raw_img = raw_img.astype(np.float32) 
    raw_save_path = img_path.replace('jpg','raw')
    raw_img.tofile(raw_save_path)
                            
    #create input list
    with open(INPUT_LIST_PATH, "w") as file: 
        file.write(raw_save_path) 

def run_model():
    print("Running inference on NPU")
    os.system(f".\qnn-net-run.exe --model quantized_mobilenet_v2.dll --backend QnnHtp.dll --input_list {INPUT_LIST_PATH} --output_dir {OUTPUT_DIR}")
    print("Inference done")

def post_process():
    print("Post-Processing the model result")
    htp_output_path = f"{OUTPUT_DIR}\Result_0\class_logits.raw"
    print("\nhtp: Result ")
    return qc_utils.postprocess(htp_output_path)
    

def process_one_image(img_path):

    pre_process(img_path)

    run_model()

    return post_process()


def run_on_webcam():

    # Open the inbuilt camera of your laptop to capture video
    cap = cv2.VideoCapture(0)
    i = 0
    frame_skip = 4  # Skip every 4th frame
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        if i % frame_skip == 0:
            frame_name = f'Frame{i}.jpg'
            cv2.imwrite(f'Frames/{frame_name}', frame)
            frame_file_path = os.path.join(FRAMES_DIR, frame_name)
            frame_output = process_one_image(frame_file_path)
            top_class = frame_output[0][0]
            top_score = round(float(frame_output[0][1]), 2)
            text = f"Frame {i}: {top_class}, {top_score}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # Display the frame
            cv2.imshow('Webcam Frame', frame)

        i += 1

        

        # Wait for a key press (0 milliseconds means indefinite wait)
        key = cv2.waitKey(1)

        # Check if the key is 'q' (ASCII value 113)
        if key == 113:
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    for image in os.listdir(FRAMES_DIR):
        image_path = os.path.join(FRAMES_DIR, image)
        os.remove(image_path)


def main():
    #img_path = r"C:\Mobilenet\kitten.jpg"
    # input_folder = r"C:\MobileNet\quant_jpg"
    # for file in os.listdir(input_folder):
    #     img_path = os.path.join(input_folder, file)
    #     process_one_image(img_path)
    
    run_on_webcam()
 

    

if __name__ == "__main__":
    main()


    


