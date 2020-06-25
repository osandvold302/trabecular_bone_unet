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
