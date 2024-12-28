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

NUM_PROC_PER_GPU = 1
NUM_GPUS = 1
BATCH_SIZE = 12
ENCODER = 'vitl'
MODEL_PATH = '/home/lab/Documents/scsv/thesis/DepthAnythingV2/checkpoints/depth_anything_v2_vitl.pth'
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
    
def compress_to_jpeg(image_array):
    """
    Compress image array to JPEG format and return as bytes
    """
    _, encoded_image = cv2.imencode('.jpg', image_array, [cv2.IMWRITE_JPEG_QUALITY, 95])
    a = torch.from_numpy(encoded_image) 
    return a

def save_depth_map(depth_map, original_size):
    """
    Convert depth map to normalized format and return both JPEG bytes and tensor
    
    Args:
        depth_map: numpy array of depth values
        original_size: tuple of (width, height)
    Returns:
        tuple: (jpeg_bytes, normalized_tensor, resized_array)
    """
    # Normalize depth map to 0-255 range
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min > 1e-6:
        normalized = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(depth_map, dtype=np.uint8)
    
    # Resize to original dimensions
    resized = cv2.resize(normalized, (original_size[0], original_size[1]), interpolation=cv2.INTER_LINEAR)
    
    # Compress to JPEG
    jpeg_bytes = compress_to_jpeg(resized)
    
    return jpeg_bytes

def process_batch(images, model, device):
    with torch.no_grad():
        out = []
        original_sizes = []
        for img in images:
            w, h = img.size
            original_sizes.append((w, h))
            nw = w + (14 - w % 14) % 14
            nh = h + (14 - h % 14) % 14
            arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            arr = cv2.resize(arr, (nw, nh))
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            out.append(t)
        return model(torch.stack(out).to(device)).cpu().numpy(), original_sizes

def process_chunk(chunk, step_path, chunk_id):
    gpu_id = chunk_id % NUM_GPUS if NUM_GPUS > 0 else None
    device = f'cuda:{gpu_id}' if gpu_id is not None else 'cpu'
    model = DepthAnythingV2(**model_configs[ENCODER])
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model = model.to(device).eval()
    
    text = f"#GPU {gpu_id} - #PROC {chunk_id}"
    results = []

    # for f in tqdm(chunk, desc=text, position=chunk_id*NUM_PROC_PER_GPU, leave=True):
    for f in chunk:
        data_file = os.path.join(step_path, f)
        data = torch.load(data_file)
        data_copy = data.copy()

        # Create output directories
        base_name = os.path.splitext(f)[0]
        depth_dir = os.path.join(os.path.dirname(data_file).replace("re10k", "re10k_depth_images"), base_name)
        os.makedirs(depth_dir, exist_ok=True)

        for idx_video in range(len(data)):

            images = [tensor_to_image(img) for img in data[idx_video]['images']]            
            depths = []            
            
            for i in tqdm(range(0, len(images), BATCH_SIZE), 
                         desc=f"File {f} - video {idx_video}/{len(data)-1} - {(chunk_id*NUM_PROC_PER_GPU)+1}, {len(images)}", 
                         position=chunk_id,                                                  
                         leave=False):                    
                batch_imgs = images[i:i+BATCH_SIZE]
                batch_depths, original_sizes = process_batch(batch_imgs, model, device)
                
                # Save each depth map in the batch
                for j, (depth_map, orig_size) in enumerate(zip(batch_depths, original_sizes)):
                    frame_idx = i + j
                    # depth_path = os.path.join(depth_dir, f'video_{idx_video}_frame_{frame_idx:04d}.jpg')
                    depth_tensor = save_depth_map(depth_map, orig_size)
                    depths.append(depth_tensor)                    

            # Update the data structure
            data_copy[idx_video]['depths'] = depths
            # data[idx_video]['depth_images'] = depth_image_tensors

        # Save updated torch file
        new_data_file = data_file.replace("re10k_test", "re10k_depth_v2")

        torch.save(data_copy, new_data_file)
        
        # results.append((f, depths))

if __name__ == '__main__':
    freeze_support()
    tqdm.set_lock(RLock())

    mp.set_start_method('spawn', force=True)
    data_path = "/home/lab/Documents/scsv/thesis/YesPoSplat/datasets/re10k_test"
    data_out = "/home/lab/Documents/scsv/thesis/YesPoSplat/datasets/re10k_depth_v2"
    
    for step in ["train", "test"]:
        step_path = os.path.join(data_path, step)
        files = [x for x in os.listdir(step_path) if x.endswith(".torch")]

        step_path_out = os.path.join(data_out, step)
        os.makedirs(step_path_out, exist_ok=True)
        
        files_out = [x for x in os.listdir(step_path_out) if x.endswith(".torch")]
        
        print("Input files:", files)
        print("Already processed:", files_out)
        files = list(set(files) - set(files_out))
        print("Files to process:", files)
        
        file_chunks = chunk_files(files, NUM_PROC_PER_GPU * NUM_GPUS)

        start_time = time.time()
        with mp.Pool(NUM_PROC_PER_GPU*NUM_GPUS) as pool:
            print(f"Processing with {NUM_PROC_PER_GPU*NUM_GPUS} workers")            
            args = [(file_chunks[i], step_path, i) for i in range(NUM_PROC_PER_GPU*NUM_GPUS)]
            print("Processing chunks:", args)
            pool.starmap(process_chunk, args)
        print(f"Processing {step} took {time.time() - start_time:.2f} seconds")