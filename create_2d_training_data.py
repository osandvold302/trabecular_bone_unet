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
from generator_viewer_3D import show_3d_images
from create_kspace_mask import gen_pdf, gen_sampling_mask, view_mask_and_pdfs




if __name__ == "__main__":

    study_dir = 'E:\\SPGR_AF2_242_242_1500_um\\'

    scan_folders = glob.glob(study_dir+'*\\data')

    # print(scan_folders)

    print('\n\n')

    num_scans = len(scan_folders)

    for counter in range(num_scans):

        print('\n\n\n ' +str(counter))

        current_scan = scan_folders[counter]

        print(current_scan)
        print(current_scan+'\\*.dcm')
        print(current_scan+'\\*.I')
        print(current_scan+'\\*.ima')

        current_scan_files = glob.glob(current_scan+'\\*.dcm')

        if len(current_scan_files) != 0:
            
            print(current_scan_files[0])

         

        # if len(current_scan_files)==0:
        #     current_scan_files =glob.glob(current_scan+'\\*.IMA')

        # if len(current_scan_files)==0:
        #     current_scan_files=glob.glob(current_scan+'\\*.ima')

        









