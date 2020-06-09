###############################
###############################
###############################
###############################
#########
#########
#########   CREATED BY: BRANDON CLINTON JONES
#########       APRIL 12, 2020
#########
#########      FOR COMPRESSED SENSING RETROSPECITVE UNDERSAMPLING OF KSPACE
#########
#########       ADAPTED FROM SIRKANT IYER'S MATLAB CODE
#########
#########
#########       Based On Miki Lustig's Monte Carlo Method
#########
#########       Publication Citation:
#########       Spare MRI: The application of CS for Rapid MRI
#########       PMID: 17969013
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
import os
from show_3d_images import show_3d_images
import numpy.fft as fft

def read_dicoms(study_dir):

    dcm_files = glob.glob(os.path.join(study_dir,'*.dcm'))
    tmp_img_fil = pydicom.dcmread(dcm_files[0])
    tmp_img = tmp_img_fil.pixel_array
    (img_height, img_width)=tmp_img.shape

    # PE_direction = tmp_img_fil.InPlanePhaseEncodingDirection 
    # print(PE_direction)

    num_slices = len(dcm_files)

    img_stack = np.zeros((img_height,img_width,num_slices),dtype=np.double)

    for counter in range(num_slices):
        tmp_img = pydicom.dcmread(dcm_files[counter])
        img_stack[:,:,counter]=tmp_img.pixel_array.astype(np.double)
        # plt.imshow(tmp_img.pixel_array)
        # plt.show()

    return img_stack


def img_to_kspace(img):
    return fft.ifftshift( fft.fftn( fft.fftshift(img)))

def kspace_to_img(kspace):
    return np.abs(fft.fftshift( fft.ifftn( fft.fftshift( kspace))))
    

if __name__ == "__main__":


    # folder = 'E:\\DL_dicom_mda_data\\SPGR_AF2_242_242_1500_um\\2012_10_22_Straus_3T\\data'

    # # NOTE: Since phase encoding is in the horiztonal direction this is what
    # # we want to subsample
    # # We don't subsample in the Readout dimension which is in 
    # # the inferior-superior direction


    # img = read_dicoms(folder)
    # # img = img / np.amax(img)*255


    # center_slice = img[:,:,30]
    
    # center_slice = center_slice[10:,:]
    # print('SLICE SHAPE')
    # print(center_slice.shape)

    # center_kspace = img_to_kspace(center_slice)

    # print('KSPACE SHAPE')
    # print(center_kspace.shape)
    # print('\n\n')

    # test_center = kspace_to_img( center_kspace )

    # print(np.amin(center_slice))
    # print(np.amax(center_slice))
    # print(np.amin(test_center))
    # print(np.amax(test_center))
    




    ############
    ############
    ############ TEST CODE FOR READING DICOM FILES IN PYTHON BELOW::
    ############
    ############


    scan_dir = (
        "E:\\DL_dicom_mda_data\\SPGR_AF2_242_242_1500_um\\2012_7_10_Moseman_3T\\data\\"
    )
    dcm_files = glob.glob(scan_dir + "*.dcm")
    # print(dcm_files)
    dcm_fil = dcm_files[0]
    img_fil = pydicom.dcmread(dcm_fil)

    print(img_fil)

    img = img_fil.pixel_array

    plt.imshow(img,cmap='gray')
    plt.show()

    

    studydate = img_fil.StudyDate
    seriesdate = img_fil.SeriesDate
    acquisitiondate = img_fil.AcquisitionDate

    series_description = img_fil.SeriesDescription
    patient_name = img_fil.PatientName

    patient_id = img_fil.PatientID
    patient_birth = img_fil.PatientBirthDate
    patient_height_meters = img_fil.PatientSize
    patient_weight = img_fil.PatientWeight
    mr_acquisition_type = img_fil.MRAcquisitionType
    sequence_name = img_fil.SequenceName
    slice_thickness = img_fil.SliceThickness
    repitition_time = img_fil.RepetitionTime
    echo_time = img_fil.EchoTime
    imaged_nucles = img_fil.ImagedNucleus
    num_averages = img_fil.NumberOfAverages
    echo_numbers = img_fil.EchoNumbers
    num_phase_encoding_steps = img_fil.NumberOfPhaseEncodingSteps
    echo_train_length = img_fil.EchoTrainLength
    pct_sampling = img_fil.PercentSampling
    pct_phase_fov = img_fil.PercentPhaseFieldOfView
    pixel_bandwidth = img_fil.PixelBandwidth
    software_version = img_fil.SoftwareVersions
    protocol_name = img_fil.ProtocolName
    transmit_coil_name = img_fil.TransmitCoilName

    in_plane_phase_encoding_direction = img_fil.InPlanePhaseEncodingDirection 
        # THIS IS THE IMPORTANT ONE


    flip_angle = img_fil.FlipAngle
    variable_flip_angle_flag = img_fil.VariableFlipAngleFlag
    sar = img_fil.SAR
    patient_position = img_fil.PatientPosition  # Ex. 'ffs' for feet first supine
    rows = img_fil.Rows
    columns = img_fil.Columns
    pixel_spacing = img_fil.PixelSpacing

    # print('\n\n\n')
    # print(img_fil.dir("hase"))

    # print(in_plane_phase_encoding_direction)

