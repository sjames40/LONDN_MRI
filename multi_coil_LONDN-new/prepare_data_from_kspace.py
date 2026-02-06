import os
import numpy as np
import time
from multiprocessing import Pool, cpu_count

# ================= configuration =================
SOURCE_KSPACE_DIR = '/egr/research-slim/hy2786/Self-Guided-DIP/NEW_KSPACE' 
OUTPUT_ROOT = './generated_dataset'

# ================= settings =================
NUM_TEST_SAMPLES = 2
ACCELERATION_FACTOR = 4 
CENTER_FRACTION = 0.08 
# ===========================================

def generate_mask(nx, ny, acc_factor, center_frac):
    mask = np.zeros((nx, ny), dtype=np.float32)
    num_cols = ny
    num_low_freqs = int(round(num_cols * center_frac))
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[:, pad:pad + num_low_freqs] = 1
    expected_lines = int(num_cols / acc_factor)
    remaining_lines = expected_lines - num_low_freqs
    all_indices = list(range(num_cols))
    outer_indices = [i for i in all_indices if i < pad or i >= pad + num_low_freqs]
    if remaining_lines > 0 and len(outer_indices) > 0:
        chosen_indices = np.random.choice(outer_indices, remaining_lines, replace=False)
        mask[:, chosen_indices] = 1
    return mask

def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x), axes=(-2, -1)), axes=(-2, -1))

# ================= core modification =================
def process_single_file(args):
    fname, source_dir, img_save_dir, mask_save_dir, acc_factor, center_frac = args
    
    try:
        file_path = os.path.join(source_dir, fname)
        
        with np.load(file_path) as data:
            k_r = data['k_r'] 
            k_i = data['k_i']
        
        kspace_full = k_r + 1j * k_i
        if kspace_full.ndim == 4: 
            kspace_full = kspace_full[0]
        
        n_coil, nx, ny = kspace_full.shape
        mask = generate_mask(nx, ny, acc_factor, center_frac)
        
        kspace_under = kspace_full * mask[np.newaxis, ...] 
        image_under = ifft2c(kspace_under)
        rss_image = np.sqrt(np.sum(np.abs(image_under)**2, axis=0))
        
        save_name = fname.replace('.npz', '.npy')
        np.save(os.path.join(img_save_dir, save_name), rss_image)
        np.save(os.path.join(mask_save_dir, save_name), mask)
        return True
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return False

def main():
    paths = {
        'train_img': os.path.join(OUTPUT_ROOT, 'four_fold_image_shape'),
        'test_img': os.path.join(OUTPUT_ROOT, 'test_four_fold'),
        'train_mask': os.path.join(OUTPUT_ROOT, '4acceleration_mask_random3'),
        'test_mask': os.path.join(OUTPUT_ROOT, '4acceleration_mask_test2')
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    print(f"Scanning file list: {SOURCE_KSPACE_DIR}")
    files = sorted(os.listdir(SOURCE_KSPACE_DIR))
    files = [f for f in files if f.endswith('.npz')]
    
    total_files = len(files)
    print(f"Found {total_files} .npz files.")
    
    if total_files == 0:
        print("No .npz files found, please check the path!")
        return

    if total_files <= NUM_TEST_SAMPLES:
        test_files = []
        train_files = files
    else:
        test_files = files[-NUM_TEST_SAMPLES:]
        train_files = files[:-NUM_TEST_SAMPLES]

    # ================= prepare task list =================
    tasks = []
    # ================= train set task =================
    for f in train_files:
        tasks.append((f, SOURCE_KSPACE_DIR, paths['train_img'], paths['train_mask'], ACCELERATION_FACTOR, CENTER_FRACTION))
    # ================= test set task =================
    for f in test_files:
        tasks.append((f, SOURCE_KSPACE_DIR, paths['test_img'], paths['test_mask'], ACCELERATION_FACTOR, CENTER_FRACTION))

    # ================= start multiprocessing =================
    # automatically detect the number of CPU cores, usually there are many cores on the server
    num_processes = max(1, cpu_count() - 2) # leave 2 cores for the system, the rest are used up
    print(f"🚀 Full throttle! Start {num_processes} processes to process {len(tasks)} tasks...")
    
    start_time = time.time()
    
    # ================= use Pool for parallel processing =================
    with Pool(processes=num_processes) as pool:
        # ================= use imap_unordered to see the progress in real time, don't wait for all to finish =================
        for i, _ in enumerate(pool.imap_unordered(process_single_file, tasks), 1):
            if i % 10 == 0:  # print progress every 10 images
                elapsed = time.time() - start_time
                speed = i / elapsed
                print(f"Progress: {i}/{len(tasks)} | Elapsed time: {elapsed:.1f}s | Speed: {speed:.1f} images/second")

    print(f"\n✅ All completed! Total time: {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    main()