import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from util.util import fft2, ifft2 

# === helper functions ===
def make_data_list(folder_path, file_list):
    full_path_list = []
    for filename in file_list:
        full_path_list.append(os.path.join(folder_path, filename))
    return full_path_list

def search_for_simliar_neighbor(test_image, training_dataset, metric, number_neighbor):
    norm_matrix2 = []
    # limit the number of iterations to prevent the search from being too slow, or set it to len(training_dataset)
    search_limit = len(training_dataset)
    
    for sample in range(search_limit):
        neighbor_path = training_dataset[sample]
        try:
            neighbor = np.load(neighbor_path)
            if neighbor.ndim == 3: neighbor = neighbor.squeeze()
            if test_image.ndim == 3: test_image_s = test_image.squeeze()
            else: test_image_s = test_image
            
            if metric == 'L1':
                distance = np.mean(np.abs(np.abs(test_image_s)-np.abs(neighbor)))
            elif metric == 'L2':
                distance = np.linalg.norm(np.abs(test_image_s)-np.abs(neighbor),'fro')
            elif metric == 'cos':
                # add a small value to prevent division by zero
                denom = (np.linalg.norm(test_image_s)*np.linalg.norm(neighbor)) + 1e-8
                distance = np.abs(np.sum(np.conj(test_image_s)*neighbor))/denom
            norm_matrix2.append(distance)
        except Exception:
            norm_matrix2.append(999999) # read failed files directly give the maximum distance

    norm_matrix2 = np.array(norm_matrix2)
    # sort and take the first N
    if len(norm_matrix2) > 0:
        if metric == 'cos':
            match_inds = np.argsort(norm_matrix2)[-number_neighbor:]
        else:
            match_inds = np.argsort(norm_matrix2)[:number_neighbor]
        return sorted(match_inds)
    else:
        return []

def make_dataset_with_output(output, dataset, kspace_data, number_neigh, metric):
    data_select = []
    mask_data_select = []
    match_inds = search_for_simliar_neighbor(output, dataset, metric, number_neigh)
    
    for b in range(len(match_inds)):
        idx = match_inds[b]
        # core fix: double boundary check to prevent Index error
        if idx < len(kspace_data) and idx < len(mask_data_set_train):
            kspace_image_data_select = kspace_data[idx]
            data_select.append(kspace_image_data_select)
            
            mask_data_set_train_data_select = mask_data_set_train[idx]
            mask_data_select.append(mask_data_set_train_data_select)
            
    return data_select, mask_data_select

# === data preparation ===
print("initializing Dataset configuration...")
Kspace_data_name = '/egr/research-slim/hy2786/Self-Guided-DIP/NEW_KSPACE'

try:
    kspace_array = sorted([f for f in os.listdir(Kspace_data_name) if f.endswith('.npz')])
except FileNotFoundError:
    print(f"path not found: {Kspace_data_name}")
    kspace_array = []

kspace_data = []
print(f"loading {len(kspace_array)} K-space files...")
# load data to memory
for j in range(len(kspace_array)):
    try:
        kspace_file = kspace_array[j]
        data = np.load(os.path.join(Kspace_data_name, kspace_file), allow_pickle=True)
        kspace_data.append(data)
    except Exception as e:
        print(f"skipping corrupted file {kspace_array[j]}: {e}")

# path definition
base_path = '/egr/research-slim/hy2786/Repo-code/LONDN_MRI/multi_coil_LONDN/generated_dataset'
image_train_data = make_data_list(os.path.join(base_path, 'four_fold_image_shape'), sorted(os.listdir(os.path.join(base_path, 'four_fold_image_shape'))))
mask_data_set_train = make_data_list(os.path.join(base_path, '4acceleration_mask_random3'), sorted(os.listdir(os.path.join(base_path, '4acceleration_mask_random3'))))
mask_data_set_test = make_data_list(os.path.join(base_path, '4acceleration_mask_test2'), sorted(os.listdir(os.path.join(base_path, '4acceleration_mask_test2'))))

# neighbor search
# modify index arbitrarily  
index = 400 
if index >= len(kspace_data): 
    print(f"index {index} out of range, reset to 0")
    index = 0

# get the current selected real file name (e.g. Kpsace_smap4352.npz)
current_file_name = kspace_array[index] 
print(f"\n[final confirmation] the current test file name is: {current_file_name}\n")

number_of_neighbor = 30
if len(kspace_data) > 0:
    print(f"reconstructing and searching for neighbors for test image (Index: {index})...")
    
    # === reconstruct a image for search ===
    target_data = kspace_data[index]
    k_r = target_data['k_r']
    k_i = target_data['k_i']
    k_complex = k_r + 1j * k_i
    # Double Shift reconstruction
    img_recon = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k_complex, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
    # magnitude merge
    test_image_abs = np.sqrt(np.sum(np.abs(img_recon)**2, axis=0))
    
    data_select, mask_data_select = make_dataset_with_output(test_image_abs, image_train_data, kspace_data, number_of_neighbor, 'L1')
else:
    print("warning: K-space data is empty")
    data_select, mask_data_select = [], []

# validation set preparation
vali_data1 = []
mask_vali = []

if len(kspace_data) > index and len(mask_data_set_test) > 0:
    vali_data1.append(kspace_data[index])
    # prevent mask from going out of bounds, if mask is not enough, use the first one
    mask_idx = index if index < len(mask_data_set_test) else 0
    mask_vali.append(mask_data_set_test[mask_idx])
else:
    print("warning: cannot build validation set")

# === Dataset class definition ===
class nyumultidataset(Dataset):
    def  __init__(self ,kspace_data,mask_data, augment=False):
        self.A_paths = kspace_data
        self.A_size = len(self.A_paths)
        self.mask_path = mask_data
        self.augment = augment
        self.nx = 640
        self.ny = 368

    def __getitem__(self, index):
        A_temp = self.A_paths[index]
        
        # 1. read original data
        s_r = A_temp['s_r']
        s_i = A_temp['s_i']
        k_r = A_temp['k_r']
        k_i = A_temp['k_i']
        
        # 2. combine to Complex Numpy
        k_np = k_r + 1j * k_i
        s_np = s_r + 1j * s_i
        
        # core fix: manual reconstruction (Double Shift)
        # 1. image reconstruction (Shift -> IFFT -> Shift)
        img_full = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k_np, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
        
        # 2. crop the size (368 -> 320)
        ncoil, nx, ny = img_full.shape
        crop_size = 320
        start_x = nx // 2 - crop_size // 2
        start_y = ny // 2 - crop_size // 2
        
        img_cropped = img_full[:, start_x:start_x+crop_size, start_y:start_y+crop_size]
        s_cropped = s_np[:, start_x:start_x+crop_size, start_y:start_y+crop_size]
        
        # 3. dynamic normalization
        mag_max = np.max(np.abs(img_cropped))
        if mag_max == 0: mag_max = 1.0
        img_norm = img_cropped / mag_max
        
        s_mag_max = np.max(np.abs(s_cropped))
        if s_mag_max == 0: s_mag_max = 1.0
        s_norm = s_cropped / s_mag_max
        
        # 4. generate Target K-space (Centered)
        k_target_complex = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_norm, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
        
        # ==================================================================
        #  key fix: manually calculate Ireal and Iunder
        # ==================================================================
        
        # --- A. generate GT (Ireal) ---
        img_gt_complex = np.sum(img_norm * np.conj(s_norm), axis=0) # [320, 320] Complex
        Ireal = torch.from_numpy(np.stack((img_gt_complex.real, img_gt_complex.imag), axis=0)).float()
        
        # --- B. generate Input (Iunder) ---
        
        if len(self.mask_path) > 0:
            # 1. try to read mask from file
            # use modulus operation, ensure that when the mask is not enough, it can be looped
            mask_file = self.mask_path[index % len(self.mask_path)]
            try:
                loaded_mask = np.load(mask_file)
                # compatible with .npz and .npy
                if isinstance(loaded_mask, np.lib.npyio.NpzFile):
                    # if it is npz, take the first array
                    mask_np = loaded_mask[loaded_mask.files[0]] 
                else:
                    mask_np = loaded_mask
                
                # ensure the mask size matches (Center Crop)
                mx, my = mask_np.shape[-2], mask_np.shape[-1]
                if mx != crop_size or my != crop_size:
                    sx = mx // 2 - crop_size // 2
                    sy = my // 2 - crop_size // 2
                    mask_np = mask_np[..., sx:sx+crop_size, sy:sy+crop_size]
                
                # convert to Tensor for later use
                # assume the input is [320, 320], we need [1, 1, 320, 320]
                prob_mask = torch.from_numpy(mask_np).float()
                if prob_mask.ndim == 2:
                    prob_mask = prob_mask.unsqueeze(0).unsqueeze(0)
                elif prob_mask.ndim == 3:
                    prob_mask = prob_mask.unsqueeze(0)
                    
            except Exception as e:
                print(f"mask read failed ({mask_file}): {e}, using random mask")
                # fallback: if read failed, generate a random mask
                prob_mask = torch.bernoulli(torch.full((1, 1, crop_size, crop_size), 0.6))
                mask_np = prob_mask.numpy().squeeze()
        else:
            # 2. no path, generate a random mask (training mode)
            prob_mask = torch.bernoulli(torch.full((1, 1, crop_size, crop_size), 0.6))
            mask_np = prob_mask.numpy().squeeze() # [320, 320]

        # ensure the type is consistent
        mask_np = mask_np.astype(np.float32)
        # core fix: end

        # apply mask to K-space
        k_under_complex = k_target_complex * mask_np
        
        # reconstruct undersampled image (Double Shift)
        img_under_coil = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k_under_complex, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
        
        # SENSE combine Input
        img_under_combined = np.sum(img_under_coil * np.conj(s_norm), axis=0)
        Iunder = torch.from_numpy(np.stack((img_under_combined.real, img_under_combined.imag), axis=0)).float()
        
        # --- C. prepare other Tensor ---
        s_tensor = torch.from_numpy(np.stack((s_norm.real, s_norm.imag), axis=1)).float()
        
        # now prob_mask must exist
        mask_return = prob_mask.squeeze(0).repeat(2, 1, 1)
        
        # force final normalization
        mag_Ireal = torch.sqrt(Ireal[0]**2 + Ireal[1]**2)
        final_scale = torch.max(mag_Ireal)
        if final_scale > 1e-6:
            Ireal = Ireal / final_scale
            Iunder = Iunder / final_scale

        # ==========================================================
        # data augmentation: random flip (Random Flip)
        # ==========================================================
        
        if torch.rand(1) < 0.5:
            Iunder = torch.flip(Iunder, dims=[-1])
            Ireal = torch.flip(Ireal, dims=[-1])
            mask_return = torch.flip(mask_return, dims=[-1])
            s_tensor = torch.flip(s_tensor, dims=[-1]) 

        if torch.rand(1) < 0.5:
            Iunder = torch.flip(Iunder, dims=[-2])
            Ireal = torch.flip(Ireal, dims=[-2])
            mask_return = torch.flip(mask_return, dims=[-2])
            s_tensor = torch.flip(s_tensor, dims=[-2])

        return  Iunder, Ireal, s_tensor, mask_return

    def __len__(self):
        return len(self.A_paths)

# === instantiate DataLoader ===
print("creating DataLoader...")
if len(data_select) == 0:
    print("warning: training set is empty, cannot continue.")
    # fake data to prevent errors
    if len(kspace_data) > 0:
        data_select = [kspace_data[0]]
        mask_data_select = [mask_data_set_train[0]] if len(mask_data_set_train) > 0 else []

train_dataset = nyumultidataset(data_select, mask_data_select, augment=True) # <--- True
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=24, 
    shuffle=True, 
    num_workers=8,     
    pin_memory=True,   
    persistent_workers=True,
    drop_last=True
)

# validation set: disable augmentation (augment=False)
test_dataset = nyumultidataset(vali_data1, mask_vali, augment=False) # <--- False
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Dataset loaded successfully (Index={index}, training set size: {len(data_select)})")
