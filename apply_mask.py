###############################
###############################
###############################
###############################
#########
#########
#########   CREATED BY: BRANDON CLINTON JONES
#########       APRIL 12, 2020
#########
#########
#########
###############################
###############################
###############################
###############################

import numpy as np
from matplotlib import pyplot as plt
import pydicom
import glob
from read_dicoms import read_dicoms, img_to_kspace, kspace_to_img
from show_3d_images import show_3d_images
from create_kspace_mask import gen_pdf, gen_sampling_mask, view_mask_and_pdfs




if __name__ == "__main__":


    folder = 'E:\\SPGR_AF2_242_242_1500_um\\2012_10_22_Straus_3T\\data'

    # NOTE: Since phase encoding is in the horiztonal direction this is what
    # we want to subsample
    # We don't subsample in the Readout dimension which is in 
    # the inferior-superior direction


    img = read_dicoms(folder)

    kspace = img_to_kspace(img)

    show_3d_images(img,upper_bound=600)

    show_3d_images(kspace)

    center_slice = np.squeeze( img[:,:,30] )
    center_kspace = img_to_kspace( center_slice )

    img_size = np.array([512,1])

    pdf , offset_value = gen_pdf(img_size=img_size, poly_order=4, usf=0.5, dist_penalty=2, radius=0.05)
    (mask, sf )=gen_sampling_mask(pdf,max_iter=150,sample_tol=0.5)

    
    view_mask_and_pdfs(the_pdf=pdf,the_mask=mask,acc_factor=sf)


    




