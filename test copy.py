from time import sleep
from tqdm import trange, tqdm
from multiprocessing import Pool, RLock, freeze_support

MAX_PROC = 2
MAX_GPU = 1

def progresser(GPU_ID, proc_id):
    interval = 0.001 / (GPU_ID + 2)
    total = 5000
    text = f"#GPU {GPU_ID} - #PROC {proc_id}"
    for _ in trange(total, desc=text, position=(MAX_PROC*GPU_ID)+proc_id, leave=True):
        sleep(interval)

if __name__ == '__main__':
    freeze_support()
    tqdm.set_lock(RLock())

    # Option 1: Using starmap
    args = [(i, j) for i in range(MAX_GPU) for j in range(MAX_PROC)]
    with Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        p.starmap(progresser, args)
