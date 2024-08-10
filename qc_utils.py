import numpy as np
from PIL import Image
import torchvision.transforms as transforms
 
def preprocess(img, nhwc = False):
    """
    Preprocesses an image for a neural network model.
 
    Args:
        img (str or PIL.Image): Path to the input image or a PIL.Image object.
        nhwc (bool, optional): If True, reshapes the output in NHWC format (channels last).
            Otherwise, uses NCHW format (channels first). Default is False.
 
    Returns:
        np.ndarray: Preprocessed image as a NumPy array.
    """
    img = Image.open(img)
    transform_fn = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform_fn(img).cpu().numpy()
    # img = img.expand_dims(axis=0)
    if(nhwc == True):
        img = img.transpose(1,2, 0)
    img = np.expand_dims(img,axis=0)
    return img
   
 
def postprocess(output,label_text_path=None):
    """
    Post-processes the output of a neural network model for image classification.
 
    Args:
        output (str or np.ndarray): The output of the neural network model. If a string(path),
                                    it is assumed to be a binary file containing the output probabilities.
        label_path (str, optional): Path to the label text file. If not provided, defaults to "synset.txt".
 
    Returns:
        None: Prints the top-5 predictions along with their corresponding class labels and probabilities.
    """
 
    if label_text_path is None:
        label_text_path = "imagenet_classes.txt"
    with open(label_text_path, 'r') as file:
        labels = [label.rstrip() for label in file]
    if isinstance(output, str):
        output = np.fromfile(output, dtype=np.float32)
        probabilities = np.exp(output) / np.sum(np.exp(output), axis=0)
    else:
        probabilities = np.exp(output[0][0]) / np.sum(np.exp(output[0][0]), axis=0)
    print("\n********************************\nTop-5 predictions:")
    scores = np.squeeze(probabilities)
    sorted_probability = np.argsort(scores)[::-1]
    output = []
    for i in sorted_probability[0:5]:
        print(' class=%s ; probability=%f100' %(labels[i],scores[i]))
        output.append((labels[i],scores[i]))
    
    return output
