import os
import io
import cv2
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.multiprocessing as mp
from multiprocessing import RLock, freeze_support
from depth_anything_v2.dpt import DepthAnythingV2

# Single model config
ENCODER = 'vitl'
MODEL_PATH = '/home/lab/Documents/scsv/thesis/DepthAnythingV2/checkpoints/depth_anything_v2_vitl.pth'
model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

BATCH_SIZE = 12  # Increase to improve GPU utilization if memory allows

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
    _, encoded_image = cv2.imencode('.jpg', image_array, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return torch.from_numpy(encoded_image)

def save_depth_map(depth_map, original_size):
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min > 1e-6:
        normalized = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(depth_map, dtype=np.uint8)

    resized = cv2.resize(normalized, (original_size[0], original_size[1]), interpolation=cv2.INTER_LINEAR)
    jpeg_bytes = compress_to_jpeg(resized)
    return jpeg_bytes

def process_batch(images, model, device):
    with torch.no_grad():
        tensors, original_sizes = [], []
        for img in images:
            w, h = img.size
            original_sizes.append((w, h))
            nw = w + (14 - w % 14) % 14
            nh = h + (14 - h % 14) % 14
            arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            arr = cv2.resize(arr, (nw, nh))
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            tensors.append(t)

        inputs = torch.stack(tensors).to(device)
        out = model(inputs).cpu().numpy()
        return out, original_sizes

def process_chunk(rank, files, step_path, data_out):
    device = torch.device(f'cuda:{rank}')
    model = DepthAnythingV2(**model_configs[ENCODER])
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model = model.to(device).eval()

    for f in files:
        data_file = os.path.join(step_path, f)
        data = torch.load(data_file)
        data_copy = data.copy()
        base_name = os.path.splitext(f)[0]

        # Output path
        depth_dir = os.path.join(
            os.path.dirname(data_file).replace("re10k", "re10k_depth_images"), 
            base_name
        )
        os.makedirs(depth_dir, exist_ok=True)

        for idx_video in range(len(data)):
            images = [tensor_to_image(img) for img in data[idx_video]['images']]
            depths = []
            for i in tqdm(range(0, len(images), BATCH_SIZE),
                          desc=f"GPU{rank} - {f} - video {idx_video}/{len(data)}",
                          position=rank,
                          leave=False):
                batch_imgs = images[i:i+BATCH_SIZE]
                batch_depths, original_sizes = process_batch(batch_imgs, model, device)
                for j, (depth_map, orig_size) in enumerate(zip(batch_depths, original_sizes)):
                    depths.append(save_depth_map(depth_map, orig_size))

            data_copy[idx_video]['depths'] = depths

        new_data_file = data_file.replace("re10k_test", "re10k_depth_v2")
        torch.save(data_copy, new_data_file)

def main():
    freeze_support()
    mp.set_start_method('spawn', force=True)
    data_path = "/home/lab/Documents/scsv/thesis/YesPoSplat/datasets/re10k"
    data_out = "/home/lab/Documents/scsv/thesis/YesPoSplat/datasets/re10k_depth"

    n_gpus = torch.cuda.device_count()

    for step in ["train", "test"]:
        step_path = os.path.join(data_path, step)
        files = [x for x in os.listdir(step_path) if x.endswith(".torch")]

        step_path_out = os.path.join(data_out, step)
        os.makedirs(step_path_out, exist_ok=True)

        files_out = [x for x in os.listdir(step_path_out) if x.endswith(".torch")]
        files = list(set(files) - set(files_out))
        file_chunks = chunk_files(files, n_gpus)

        start_time = time.time()
        processes = []
        for rank in range(n_gpus):
            print(f"Starting process {rank}")
            p = mp.Process(
                target=process_chunk,
                args=(rank, file_chunks[rank], step_path, data_out)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print(f"Processing {step} took {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
