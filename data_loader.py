###############################
###############################
###############################
###############################
#########
#########
#########   CREATED BY: BRANDON CLINTON JONES
#########       APRIL 20, 2020
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
from scipy.io import loadmat
import numpy.fft as fft
import numpy.matlib

#############
#############
#############       CUSTOM DEPENDENCIES!!!!!
#############
#############


from show_3d_images import show_3d_images
from create_kspace_mask import gen_pdf, gen_sampling_mask, view_mask_and_pdfs

def create_aliased_image(input_mat, acceleration_factor=4):
    # NOTE: Takes stack of input 2d images OR stack of 2d kspace
    # It then subsamples the kspace uniformly based on the acceleration factor

    (img_height, img_width, num_images) = input_mat.shape

    if np.any(np.iscomplex(input_mat)):  # If the matrix is complex do nothin
        kspace = input_mat
    else:  # IF the image is not complex, turn it into kspace
        kspace = np.empty(input_mat.shape, dtype=np.cdouble)

        for counter in range(num_images):
            kspace[:, :, counter] = img_to_kspace(input_mat[:, :, counter])

    # Now sample the center of kspace
    # Only downsampling in the width direction since readout is vertically

    mask_1d = np.zeros((1, img_width))

    image_aliased = np.zeros(kspace.shape, dtype=np.double)
    mask_3d = np.zeros(kspace.shape, dtype=np.bool)

    if (img_width % 2) == 0:
        # This is even matrix
        # Sample the center two matrix points
        half_size = int(img_width / 2)

        mask_1d[:, half_size - 1] = 1
        mask_1d[:, half_size] = 1
    else:
        # Odd matrix. Sample only center point
        half_size = np.floor(img_width / 2).astype(np.int16)
        mask_1d[:, half_size] = 1

    for counter in range(num_images):
        mask_1d_tmp = np.copy(mask_1d)
        rand_start = np.random.randint(low=0, high=acceleration_factor - 1)
        indices = np.arange(
            start=rand_start, stop=img_width - 1, step=acceleration_factor
        )
        mask_1d_tmp[:, indices] = 1
        mask_2d = np.matlib.repmat(mask_1d_tmp, img_height, 1)

        undersample_ratio = np.sum(mask_1d_tmp.flatten()) / img_width * 100
        del mask_1d_tmp

        mask_3d[:, :, counter] = mask_2d
        image_aliased[:, :, counter] = kspace_to_img(
            np.multiply(kspace[:, :, counter], mask_2d)
        )

    return image_aliased, mask_3d, undersample_ratio


def load_data(study_dir, is_2d=True):

    print("Loading images and kspace . . . \n\n")
    print(study_dir + "*\\data\\dataFile.mat")
    scan_files = glob.glob(study_dir + "*\\data\\dataFile.mat")

    num_scans = len(scan_files)

    for counter in range(num_scans):

        img = loadmat(scan_files[counter])

        img = img["img"]

        if counter == 0:
            (img_height, img_width, num_slices) = img.shape
            if is_2d:
                img_stack = np.zeros(
                    (img_height, img_width, num_scans), dtype=np.double
                )
                kspace_stack = np.zeros(
                    (img_height, img_width, num_scans), dtype=np.cdouble
                )


        if is_2d:
            img_stack[:, :, counter] = img[:, :, 29]
            kspace_stack[:, :, counter] = img_to_kspace(np.squeeze(img[:, :, 29]))


    return img_stack, kspace_stack


def img_to_kspace(img):
    return fft.ifftshift(fft.fftn(fft.fftshift(img)))


def kspace_to_img(kspace):
    return np.abs(fft.fftshift(fft.ifftn(fft.fftshift(kspace)))).astype(np.double)


def mask_kspace(
    full_kspace,
    acceleration_factor,
    polynomial_order=7,
    distance_penalty=2,
    center_maintained=0,
):

    print("Undersampling kspace . . . \n\n")
    img_height, img_width, num_slices = full_kspace.shape
    undersampling_factor = 1.0 / acceleration_factor

    undersampled_kspace = np.zeros(full_kspace.shape, dtype=np.cdouble)
    mask_3d = np.zeros(full_kspace.shape, dtype=np.bool)

    for counter in range(num_slices):

        (pdf, offset_value) = gen_pdf(
            img_size=(1, img_width),
            poly_order=polynomial_order,
            usf=undersampling_factor,
            dist_penalty=distance_penalty,
            radius=center_maintained,
        )

        (mask_1d, sf) = gen_sampling_mask(pdf, max_iter=150, sample_tol=1.5)

        mask_2d = np.matlib.repmat(mask_1d, img_height, 1).astype(np.cdouble)
        # Take the horiztonal 1D kspace psueorandom mask and replicate it down vertically
        # To create the actual 2D kspace mask that will be used for the data
        #       NOTE: Replicated vertically since we have no penalty in the readout direction
        #       so we don't subsample in readout

        mask_3d[:, :, counter] = mask_2d

        # undersampled_slice=np.multiply(mask,full_kspace[:,:,counter])
        undersampled_slice = np.multiply(full_kspace[:, :, counter], mask_2d)
        undersampled_kspace[:, :, counter] = undersampled_slice

    return undersampled_kspace, mask_3d


def generate_data(the_study_dir, speed_factor=2):

    (images, kspace_full) = load_data(study_dir=the_study_dir)

    kspace_undersampled, kspace_3d_mask = mask_kspace(
        full_kspace=kspace_full, acceleration_factor=speed_factor
    )

    return images, kspace_full, kspace_undersampled, kspace_3d_mask


if __name__ == "__main__":

    the_study_dir = "E:\\ENM_Project\\SPGR_AF2_242_242_1500_um\\"

    # NOTE: Since phase encoding is in the horiztonal direction this is what
    # we want to subsample
    # We don't subsample in the Readout dimension which is in
    # the inferior-superior direction

    af = 4

    (images, kspace_full, kspace_undersampled, kspace_3d_mask) = generate_data(
        the_study_dir=the_study_dir,
        speed_factor=af
    )

    (aliased_images, aliased_mask, aliad_undersampling_ratio) = create_aliased_image(
        images, acceleration_factor=af
    )

    kspace_aliased = np.multiply(kspace_full, aliased_mask)

    img_height, img_width, num_slices = images.shape

    test_downsampled_img = np.zeros(images.shape)
    for cc in range(num_slices):
        test_downsampled_img[:, :, cc] = kspace_to_img(kspace_undersampled[:, :, cc])

    plot_upper_bound = 500
    full_kspace_mask = np.ones((images.shape))



    plot_images = np.hstack((images, test_downsampled_img, aliased_images))
    plot_kspace_masks = (
        np.hstack((full_kspace_mask, kspace_3d_mask, aliased_mask)) * plot_upper_bound
    )

    show_3d_images(
        np.vstack((plot_images, plot_kspace_masks)), upper_bound=plot_upper_bound
    )


    show_3d_images(np.hstack((kspace_full, kspace_undersampled, kspace_aliased)))

    # scan_num = 1
    # img_plot = np.vstack((plot_images[:,:,scan_num], plot_kspace_masks[:,:,scan_num]))
    # # fig1, axes1 = plt.subplots(1,2)
    # plt.imshow(img_plot,cmap='gray',vmin=0,vmax=plot_upper_bound)
    # save_name='img1_af_'+str(af)+'.png'
    # plt.savefig('D:\\Desktop\\RESEARCH\\Lab_Meetings\\Research_Presentations\\'+save_name)
    # plt.show()

    

    # kspace_plot=np.hstack((kspace_full[:,:,scan_num], 
    #     kspace_undersampled[:,:,scan_num], 
    #     kspace_aliased[:,:,scan_num]))   
    

    # kspace_plot = np.log( np.abs( kspace_plot)+1)
    # kspace_plot=kspace_plot-np.amin(kspace_plot)
    # kspace_plot = kspace_plot/np.amax(kspace_plot)

    # plt.imshow(kspace_plot,cmap='gray',vmin=0,vmax=1)
    # save_name='kspace1_af_'+str(af)+'.png'
    # plt.savefig('D:\\Desktop\\RESEARCH\\Lab_Meetings\\Research_Presentations\\'+save_name)
    # plt.show()

    