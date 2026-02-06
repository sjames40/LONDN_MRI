import sys
import os
import time
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models import networks
from util.metrics import PSNR
import local_network_dataset
from models.Unet_model_fast_mri import Unet
from util.util import convert_2chan_into_complex, init_weights

# ================= configuration region =================
# import dataset module to get the index variable
import local_network_dataset 

# get the index of the current test image
current_idx = local_network_dataset.index

# import regular expression
import re

# 1. get the full file name from the dataset (e.g. 'Kpsace_smap4352.npz')
file_name_full = local_network_dataset.current_file_name

# 2. extract pure number ID
numbers = re.findall(r'\d+', file_name_full)
file_id = numbers[-1] if numbers else "unknown"

# 3. name the folder with the ID (e.g. ./checkpoints/exp_4352)
SAVE_DIR = f'./checkpoints/exp_{file_id}'

print(f"file name extracted successfully: {file_name_full} -> ID: {file_id}")
print(f"folder automatically located at: {SAVE_DIR}")

EPOCHS = 600
LEARNING_RATE = 2e-4
CG_ITER = 6
REGU_PARAM = 1e-10

# create save folder
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"folder created: {SAVE_DIR}")

# initialize log file (CSV format, convenient to open with Excel)
log_path = os.path.join(SAVE_DIR, 'train_log.csv')
if not os.path.exists(log_path):
    with open(log_path, 'w') as f:
        f.write('Epoch,Train_Loss,Val_Loss,PSNR,Time\n') # write header

# initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# initialize UNet
netG = Unet(2, 2, num_pool_layers=2, chans=64).to(device)
init_weights(netG, init_type='normal', init_gain=0.01)
netG = netG.float()

fn = nn.MSELoss().to(device)
optimG = torch.optim.Adam(netG.parameters(), lr=LEARNING_RATE, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimG, milestones=[100, 200], gamma=0.65)

def CG(output, tol, L, smap, mask, alised_image):
    return networks.CG.apply(output, tol, L, smap, mask, alised_image)

print("start training...")

# record the best PSNR
best_psnr = 0.0

# use tqdm to create progress bar
pbar = tqdm(range(EPOCHS), desc="Training", unit="epoch")

# === debug probe: only run before the first loop, check the data range ===
try:
    first_batch = next(iter(local_network_dataset.train_loader))
    d_in, d_target, _, _ = first_batch
    print(f"\n [DEBUG data check]")
    print(f"   Input Range: Min={d_in.min().item():.5f}, Max={d_in.max().item():.5f}")
    print(f"   Label Range: Min={d_target.min().item():.5f}, Max={d_target.max().item():.5f}")
    if d_in.max().item() > 100 or d_in.max().item() < 0.001:
        print("warning: data range may be abnormal, please check normalization!")
    print("-" * 30 + "\n")
except Exception as e:
    print(f"DEBUG probe failed (does not affect training): {e}")
# ================================================

for epoch in pbar:
    epoch_start_time = time.time()
    loss_G_train = 0.0
    
    # ------------------ training loop ------------------
    netG.train() # ensure entering training mode
    train_steps = 0
    for direct, target, smap, mask in local_network_dataset.train_loader:    
        noise_input = direct.to(device).float()
        smap = smap.to(device).float()
        mask = mask.to(device).float()
        label = target.to(device).float()
        
        temp = noise_input
        # CG iteration
        for ii in range(CG_ITER):
            output = netG(temp)
            output2 = CG(output, tol=0.00001, L=1, smap=smap, mask=mask, alised_image=noise_input).type(torch.float32)
            temp = output2
            
        output_final = temp
        
        optimG.zero_grad()
        loss_G = fn(output_final, label)
        
        # L1 regularization
        l1_regularization = torch.tensor(0., device=device)
        for param in netG.parameters():
           l1_regularization += torch.norm(param, 1)
           
        total_loss = loss_G + REGU_PARAM * l1_regularization
        total_loss.backward()
        # add this line: gradient clipping
        torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
        optimG.step()

        loss_G_train += total_loss.item()
        train_steps += 1
        
    # calculate average training loss
    avg_train_loss = loss_G_train / train_steps if train_steps > 0 else 0

# ------------------ validation loop ------------------
    netG.eval() # enter evaluation mode
    val_loss_sum = 0.0
    psnr_sum = 0.0
    val_steps = 0
    
    with torch.no_grad():
        for vali_direct, vali_target, vali_smap, vali_mask in local_network_dataset.test_loader:
            vali_input = vali_direct.to(device).float()
            vali_smap = vali_smap.to(device).float()
            vali_mask = vali_mask.to(device).float()
            vali_label = vali_target.to(device).float()
            
            vali_temp = vali_input
            # CG iteration
            for jj in range(CG_ITER):
                vali_output = netG(vali_temp)
                vali_output2 = CG(vali_output, tol=0.00001, L=1, smap=vali_smap, mask=vali_mask, alised_image=vali_input).type(torch.float32)
                vali_temp = vali_output2
            
            vali_result = vali_temp
            
            # calculate validation loss
            v_loss = fn(vali_result, vali_label)
            # note: validation set usually does not need L1 regularization loss, only look at the reconstruction loss
            val_loss_sum += v_loss.item()
            
            # 1. convert to numpy
            pred_np = vali_result.cpu().detach().numpy() # [1, 2, H, W]
            gt_np = vali_label.cpu().detach().numpy()    # [1, 2, H, W]
            
            # 2. synthesize magnitude image (Magnitude)
            img_pred = np.sqrt(pred_np[:, 0, :, :]**2 + pred_np[:, 1, :, :]**2)
            img_gt = np.sqrt(gt_np[:, 0, :, :]**2 + gt_np[:, 1, :, :]**2)
            
            # 3. remove Batch dimension -> become 2D image [H, W]
            img_pred = img_pred.squeeze()
            img_gt = img_gt.squeeze()

            # 4. ensure no NaN (Not a Number)
            img_pred = np.nan_to_num(img_pred)
            img_gt = np.nan_to_num(img_gt)

            # force truncation to [0, 1] range
            img_pred = np.clip(img_pred, 0, 1)
            img_gt = np.clip(img_gt, 0, 1)

            # 5. manually calculate PSNR
            mse = np.mean((img_gt - img_pred) ** 2)
            if mse < 1e-10: # prevent division by 0
                current_psnr = 100
            else:
                max_pixel = 1.0
                current_psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

            psnr_sum += current_psnr
            val_steps += 1
            
            # ==========================================================
            # plot debugging
            # ==========================================================
            if epoch % 10 == 0 and val_steps == 1:
                try:
                    import matplotlib.pyplot as plt
                    
                    # calculate the Input image for comparison
                    t_in = vali_input[0].cpu().detach().numpy() 
                    img_in = np.sqrt(t_in[0]**2 + t_in[1]**2) 
                    
                    plt.figure(figsize=(12, 4))
                    plt.subplot(131); plt.imshow(img_in, cmap='gray'); plt.title('Input (Undersampled)')
                    plt.subplot(132); plt.imshow(img_pred, cmap='gray'); plt.title(f'Output (Epoch {epoch})') # use new variable
                    plt.subplot(133); plt.imshow(img_gt, cmap='gray'); plt.title('GT') # use new variable
                    
                    plt.savefig(os.path.join(SAVE_DIR, f'debug_epoch_{epoch}.png'))
                    plt.close()
                except Exception as e:
                    print(f"plot failed (does not affect training): {e}")
            # ==========================================================

    scheduler.step()
    
    # ------------------ statistics and display ------------------
    avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0
    avg_psnr = psnr_sum / val_steps if val_steps > 0 else 0
    
    # 1. update the bottom progress bar (only show the current status)
    pbar.set_postfix({
        'T-Loss': f'{avg_train_loss:.5f}', 
        'PSNR': f'{avg_psnr:.2f}dB'
    })

    # 2. write to CSV log
    with open(log_path, 'a') as f:
        f.write(f'{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f},{avg_psnr:.4f},{datetime.datetime.now().strftime("%H:%M:%S")}\n')

    # 3. print the log to the screen (this line will be retained on the screen, will not disappear)
    tqdm.write(f"Epoch {epoch+1:03d} | T-Loss: {avg_train_loss:.6f} | V-Loss: {avg_val_loss:.6f} | PSNR: {avg_psnr:.4f} dB")

    # 4. save model strategy
    # A. save the latest model (Latest)
    torch.save(netG.state_dict(), os.path.join(SAVE_DIR, 'latest_net_G.pth'))
    
    # B. save the best model (Best PSNR)
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(netG.state_dict(), os.path.join(SAVE_DIR, 'best_net_G.pth'))
        tqdm.write(f"new record, PSNR: {avg_psnr:.2f} dB -> saved best_net_G.pth")
    
    # C. backup every 50 epochs
    if (epoch + 1) % 50 == 0:
        torch.save(netG.state_dict(), os.path.join(SAVE_DIR, f'net_G_epoch_{epoch+1}.pth'))

print(f"\n training completed, best PSNR: {best_psnr:.2f} dB")
