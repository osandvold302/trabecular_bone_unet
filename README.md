# trabecular_bone_unet
Rajapakse/Jones project for reconstructing artificially undersampled kspace MRI scans of hips. Based on work completed by bcjones to reconstruct 2D images.  
  
## Background Motivation  
MRI alternative for measuring bone density and predicting fractures over CT -> reduction in radiation dose, high res images. Time to 
collect longer than practical, so undersampling k-space (acquisition domain) to shorten period. Leads to artifacts within images. 
Need to reconstruct images to accurately assess bone properities.  
  
## Completed 
The initial state of the codebase demonstrates it is possible to train a U-Net (modified CNN) to reconstruct 2D slices artificially 
undersampled to produce similar measurements of fully sampled counterparts. Training completed on images generated from undersampled 
k-space. 
  
Images -> k-space -> PDF generates random sampling xINT -> selection of least interface kspace -> image with artifacts (input to CNN)  
  
## Goal
Apply to 3D images

## Usage  
There are currently 2 command line options and many in-code configurations 
you can make for training this network.  

### Command Line  
Supplying the `--3d` flag will tell the network to use the full 3d volume 
for each scan. Supplying `--2d` or no flag at all will run the network on 
2d images (the center slice)  

Providing `-d <path_to_save_dir>` will save every 20th model and the last 
generated model to the relative `path_to_save_dir`. If you choose not to 
provide a path, then the models will *NOT be saved!* If the path does not 
exist, then it will be created during runtime. Failure to create this 
directory will result in terminating the process.  

### In-Code Parameters  
Right now, these are in `main.py` and `data_loader_class.py`.  

In `main.py`, we have acceleration factor, batch size, learning rate, 
polyfit, and max number of epochs.  
In `data_loader_class.py`, we have the data directory and files of 
interest and how many iterations should be run to find the ideal 
undersampling mask.

