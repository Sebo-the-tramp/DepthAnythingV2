from time import sleep
from tqdm import trange, tqdm

import io
import os
import cv2
import time
import torch
import torch.multiprocessing as mp
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from depth_anything_v2.dpt import DepthAnythingV2

from multiprocessing import Pool, RLock, freeze_support

NUM_PROC_PER_GPU = 2
NUM_GPUS = 1

BATCH_SIZE = 2
ENCODER = 'vitl'
MODEL_PATH = 'checkpoints/depth_anything_v2_vitl.pth'
model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

def chunk_files(files, n):
    size = len(files) // n
    remainder = len(files) % n
    chunks, start = [], 0
    for i in range(n):
        end = start + size + (1 if i < remainder else 0)
        chunks.append(files[start:end])
        start = end
    return chunks

def tensor_to_image(t):
    b = t.cpu().numpy().tobytes()
    s = io.BytesIO(b)
    try:
        img = Image.open(s)
        img.load()
        return img
    except:
        return None
    
def depth_to_image_tensor(depth_map):
    """
    Convert a single depth map (H,W) into a normalized torch.uint8 tensor,
    or store raw float values as you prefer.
    """
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min < 1e-6:
        norm = np.zeros_like(depth_map, dtype=np.uint8)
    else:
        norm = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    return torch.from_numpy(norm)  # shape [H, W] in uint8

def process_batch(images, model, device):
    with torch.no_grad():
        out = []
        for img in images:
            w, h = img.size
            nw = w + (14 - w % 14) % 14
            nh = h + (14 - h % 14) % 14
            arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            arr = cv2.resize(arr, (nw, nh))
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            out.append(t)
        return model(torch.stack(out).to(device)).cpu().numpy()

def process_chunk(chunk, step_path, chunk_id):

    gpu_id = chunk_id % NUM_GPUS if NUM_GPUS > 0 else None
    device = f'cuda:{gpu_id}' if gpu_id is not None else 'cpu'
    model = DepthAnythingV2(**model_configs[ENCODER])
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model = model.to(device).eval()
    
    text = f"#GPU {gpu_id} - #PROC {chunk_id}"
    results = []

    for f in tqdm(chunk, desc=text, position=chunk_id*NUM_PROC_PER_GPU, leave=True):
        data_file = os.path.join(step_path, f)
        data = torch.load(data_file)

        for idx_video in range(len(data)-1):             
            with ThreadPoolExecutor() as ex:
                images = list(ex.map(tensor_to_image, data[idx_video]['images']))
            depths = []
            for i in tqdm(range(0, len(images), BATCH_SIZE), desc=f"File {f} - video {idx_video}/{len(data)-1} - {(chunk_id*NUM_PROC_PER_GPU)+1}", position=(chunk_id*NUM_PROC_PER_GPU)+1, leave=False):
                batch_imgs = images[i:i+BATCH_SIZE]
                depths.extend(process_batch(batch_imgs, model, device))

            # Save to original .torch data structure
            data[idx_video]['depths'] = depths
            # Convert depth maps to normalized 
            # images, then store them:
            depth_image_tensors = []
            for d in depths:
                depth_image_tensors.append(depth_to_image_tensor(d))
            data[idx_video]['depth_images'] = depth_image_tensors

        # Save back to the .torch file              
        new_data_file = data_file.replace("re10k", "re10k_depth")        
        torch.save(data, new_data_file)
        
        results.append((f, depths))
    # for _ in trange(total, desc=text, position=(MAX_PROC*GPU_ID)+proc_id, leave=True):
    #     pass        

if __name__ == '__main__':
    freeze_support()
    tqdm.set_lock(RLock())

    mp.set_start_method('spawn', force=True)
    data_path = "../YesPoSplat/datasets/re10k"
    data_out = "../YesPoSplat/datasets/re10k_depth"

    for step in ["train", "test"]:
        step_path = os.path.join(data_path, step)
        files = [x for x in os.listdir(step_path) if x.endswith(".torch")]

        step_path_out = os.path.join(data_out, step)
        files_out = [x for x in os.listdir(step_path_out) if x.endswith(".torch")]

        print(files)
        print(files_out)
        files = list(set(files) - set(files_out))
        print(files) 
        file_chunks = chunk_files(files, NUM_PROC_PER_GPU * NUM_GPUS)

        with mp.Pool(NUM_PROC_PER_GPU*NUM_GPUS) as pool:
            print(NUM_PROC_PER_GPU*NUM_GPUS)            
            args = [(file_chunks[i], step_path, i) for i in range(NUM_PROC_PER_GPU*NUM_GPUS)]
            print(args)
            for result in pool.starmap(process_chunk, args):
                # i should save the results here to the original torch file
                pass
