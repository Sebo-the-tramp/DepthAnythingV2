# The idea here is to get the images, and the views, and compute the depth map for each with like depth anythng or something similar

# I don't know then how to merge the two but I think probably that is done later, there is nothing already computed in 3D.

import io
import os
import cv2
import torch

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

def tensor_to_image(tensor):
    """
    Converts a PyTorch tensor containing JPEG byte data to a PIL Image.
    
    Args:
        tensor (torch.Tensor): A 1D tensor of type torch.uint8 representing JPEG bytes.
    
    Returns:
        PIL.Image.Image: The decoded image.
    """
    # Ensure the tensor is on CPU and convert to NumPy array
    byte_array = tensor.cpu().numpy().tobytes()
    
    # Use BytesIO to handle the byte data
    image_stream = io.BytesIO(byte_array)
    
    # Open the image using PIL
    try:
        image = Image.open(image_stream)
        # Ensure the image is loaded before closing the stream
        image.load()
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
def rgb_to_depth(rgb_images):

    for image in rgb_images:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # resize image to be divisible by 14
        h, w = image.shape[:2]
        new_h = h + (14 - h % 14) % 14
        new_w = w + (14 - w % 14) % 14
        print(new_h, new_w)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            depth = model(image.to(DEVICE)).cpu().numpy().squeeze()

        plt.imshow(depth, cmap='magma')    

    pass

def main():
    data_path = "../YesPoSplat/datasets/re10k"

    for step in ["train", "test"]:

        files = os.listdir(data_path + "/" + step)
        for file in files:
            if file.endswith(".torch"):
                print(file)
                data = torch.load(data_path + "/" + step + "/" + file)
                decoded_images = [tensor_to_image(tensor) for tensor in data[0]['images']]

                rgb_to_depth(decoded_images)

                print(len(decoded_images))


if __name__ == "__main__":
    main()