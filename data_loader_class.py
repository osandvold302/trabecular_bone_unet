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
import os
import tensorflow as tf
import argparse
import logging

#############
#############
#############       CUSTOM DEPENDENCIES!!!!!
#############
#############


from show_3d_images import show_3d_images
from create_kspace_mask import gen_pdf, gen_sampling_mask, view_mask_and_pdfs

def get_logger(name):
    log_format = "%(asctime)s %(name)s %(levelname)5s %(message)s"
    logging.basicConfig(level=logging.DEBUG,format=log_format,
                        filename='dev.log',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(name).addHandler(console)
    logging.basicConfig(filename='LOGFILE.log',filemode='w')

    return logging.getLogger(name)

class data_generator:
    def __init__(
        self,
        #study_dir="/d1/hip/DL/SPGR_AF2_242_242_1500_um/", # on LSNI2
        study_dir="./SPGR_AF2_242_242_1500_um/",
        valid_ratio=0.2,  # Percent of total data that is validation. Train is 1-Ratio
        batch_size=10,
        acceleration_factor=2,
        bool_2d=True,
        poly_distance_order=4,  # Polynomial fit for L1/L2 distance matrix
        distance_penalty=2,  # L1 or L2. But it must be L1 for 2D case iirc
        center_maintained=0,  # Percent of center of kspace that must be sampled
        # If 0.1, center 10% is ALWAYS sampled
        bool_shuffle=True,
        create_new_dataset=False,
    ):

        super(data_generator, self).__init__()

        self.study_dir = study_dir

        # self.standardize_data = True
        self.standardize_data = True

        self.valid_ratio = valid_ratio
        self.acceleration_factor = acceleration_factor
        self.batch_size = batch_size
        self.is_2d = bool_2d
        self.polynomial_order = poly_distance_order
        self.distance_penalty = distance_penalty
        self.center_maintained = center_maintained
        self.data_type = tf.complex64
        self.bool_shuffle = bool_shuffle

        # Will have dimensions (94, 1, 512, 512) or (94, 1, 512, 512, 60) depending on 2d/3d
        (images_full, kspace_full, patient_names) = self.load_data()
        self.split_train_and_valid(images_full, kspace_full, patient_names)


    def load_data(self):
        print("\nLoading images and kspace . . . \n")

        study_dir = self.study_dir
        
        #lsni
        study_dir = '/d1/hip/DL/SPGR_AF2_242_242_1500_um/'

        # scan_fils = glob.glob(study_dir + "*\\data\\dataFile.mat")
        scan_files = glob.glob(study_dir + "2012*/data/dataFile.mat")
        
        # Used for cluster
        #scan_files = glob.glob(study_dir + "12*dataFile.mat")

        scan_name_list = []

        # Save the Patient ID to a list of the same order as the data will be loaded
        for counter in range(len(scan_files)):
            '''Used on LSNI
            (_, split_drive) = os.path.splitdrive(scan_files[counter])
            (data_folder, _) = os.path.split(split_drive)
            (scan_folder, _) = os.path.split(data_folder)
            (_, scan_name) = os.path.split(scan_folder)
            '''
            # Used for the cluster
            fileName = os.path.basename(scan_files[counter])
            scan_name = fileName.split('_')[0]
            # TODO: Create flag for switching between
            scan_name_list.append(scan_name)

        num_scans = len(scan_files)
        print("num scans:" + str(num_scans))

        for counter in range(num_scans):

            img = loadmat(scan_files[counter])

            # img = img["img"]
            img = img["D"] # NOTE: matfile has "D" key not "img"

            if counter == 0:
                (img_height, img_width, num_slices) = img.shape
                if self.is_2d:
                    # NOTE: OUTPUT SHOULD HAVE SHAPE
                    # (BATCH_DIM, CHANNEL_DIM, IMG_HEIGHT, IMG_WIDTH)
                    # (NUM_SCANS, CHANNEL_DIM, IMG_HEIGHT, IMG_WIDTH)
                    # (94, 1, 512, 512)

                    img_stack = np.zeros(
                        (num_scans, 1, img_height, img_width), dtype=np.double
                    )
                    kspace_stack = np.zeros(
                        (num_scans, 1, img_height, img_width), dtype=np.cdouble
                    )

                else:
                    # NOTE: OUTPUT HAS SHAPE
                    # (BATCH_DIM, CHANNEL_DIM, IMG_HEIGHT, IMG_WIDTH, NUM_SLICES)
                    # (NUM_SCANS, CHANNEL_DIM, IMG_HEIGHT, IMG_WIDTH, NUM_SLICES)
                    # (94, 1, 512, 512, 60)

                    img_stack = np.zeros(
                        (num_scans, 1, img_height, img_width, num_slices), dtype=np.double
                    )
                    kspace_stack = np.zeros(
                        (num_scans, 1, img_height, img_width, num_slices), dtype=np.cdouble
                    )


            if self.is_2d:
                if self.standardize_data:
                    tmp_img = img[:, :, 29]
                    tmp_img = tmp_img - np.amin(tmp_img)
                    tmp_img = tmp_img / np.amax(tmp_img)
                    img_stack[counter, 0, :, :] = tmp_img
                    kspace_stack[counter, 0, :, :] = self.img_to_kspace(tmp_img)

                else:
                    img_stack[counter, 0, :, :] = img[:, :, 29]
                    kspace_stack[counter, 0, :, :] = self.img_to_kspace(img[:, :, 29])
            else:
                img_stack[counter, 0, :, :, :] = img
                kspace_stack[counter, 0, :, :, :] = self.img_to_kspace(img[:,:, :])

        return img_stack, kspace_stack, scan_name_list

    def split_train_and_valid(self, images, kspace, names_list):
        print("\nSplitting Training And Valid Data . . . \n")

        total_images = len(names_list) # assumption that names = total num
        num_valid = np.round(self.valid_ratio * total_images).astype(np.int16)
        num_train = int(total_images - num_valid)
        
        total_arange = np.arange(total_images) # total_arange = indices
        np.random.shuffle(total_arange)
        total_arange = total_arange.astype(np.int16)
        # Randomize the order of the data set

        names_list = np.array(names_list)
        names_list = names_list[total_arange]
        names_list = list(names_list)

        if self.is_2d:

            images = images[total_arange, :, :, :]
            kspace = kspace[total_arange, :, :, :]

            # Now split into training and validation data
            train_names = names_list[:num_train]
            train_images = images[:num_train, :, :, :]
            train_kspace = kspace[:num_train, :, :, :]

            valid_names = names_list[num_train + 1 :]
            valid_images = images[num_train + 1 :, :, :, :]
            valid_kspace = kspace[num_train + 1 :, :, :, :]

            self.train_names = train_names
            self.train_images = train_images
            self.train_kspace = train_kspace

            self.valid_names = valid_names
            self.valid_images = valid_images
            self.valid_kspace = valid_kspace

        else:
            images = images[total_arange, :, :, :, :]
            kspace = kspace[total_arange, :, :, :, :]

            # Now split into training and validation data
            train_names = names_list[:num_train]
            train_images = images[:num_train, :, :, :, :]
            train_kspace = kspace[:num_train, :, :, :, :]

            valid_names = names_list[num_train + 1 :]
            valid_images = images[num_train + 1 :, :, :, :, :]
            valid_kspace = kspace[num_train + 1 :, :, :, :, :]

            self.train_names = train_names
            self.train_images = train_images
            self.train_kspace = train_kspace

            self.valid_names = valid_names
            self.valid_images = valid_images
            self.valid_kspace = valid_kspace


    def get_batch(self):
        print("\n Getting batch . . . \n")

        if self.is_2d:

            (num_train, num_channels, img_height, img_width) = self.train_kspace.shape
            (num_valid, _, _, _) = self.valid_kspace.shape

            return (
                num_train,
                num_valid,
                num_channels,
                img_height,
                img_width,
                self.train_names,
                self.valid_names,
                # self.kspace_mean,
                # self.kspace_std
            )

    def get_info(self):
        print("\n Returning Dataset Info . . . \n")
        num_train = num_channels = img_height = img_width = num_slices = 0

        if self.is_2d:
            (num_train, num_channels, img_height, img_width) = self.train_kspace.shape
            (num_valid, _, _, _) = self.valid_kspace.shape

            return (
                num_train,
                num_valid,
                num_channels,
                img_height,
                img_width,
                self.train_names,
                self.valid_names,
                # self.kspace_mean,
                # self.kspace_std
            )
        else:
            (num_train, num_channels, img_height, img_width, num_slices) = self.train_kspace.shape
            (num_valid, _, _, _, _) = self.valid_kspace.shape
            
            return (
                num_train,
                num_valid,
                num_channels,
                img_height,
                img_width,
                num_slices,
                self.train_names,
                self.valid_names,
                # self.kspace_mean,
                # self.kspace_std
            )


    def generator(
        self, batch_ind, is_train=True, is_image_space=True, return_masks=True
    ):

        if batch_ind == 0 and self.bool_shuffle:
            # print('SHUFFLING \n\n')
            self.shuffle_data()

        if is_train:
            kspace_matrix = self.train_kspace
        else:
            kspace_matrix = self.valid_kspace

        if self.is_2d:

            # NOTE: IMGAES ARE SAVED AS
            #  (NUM_TRAIN, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH )

            (total_images, channel_dim, height_dim, width_dim) = kspace_matrix.shape

            steps_per_epoch = int(np.ceil(total_images / self.batch_size))

            # for counter in range(steps_per_epoch):
            # Images and Kspace matrices are of the order
            # Height, Width, Scan_slice
            # Readout, Phase Encoding, Center Slice of scan

            start_ind = batch_ind * self.batch_size
            end_ind = start_ind + self.batch_size

            if end_ind > total_images:
                end_ind = total_images
                start_ind = total_images - self.batch_size

            kspace_batch_full = kspace_matrix[start_ind:end_ind, :, :, :]

            # NOTE: GENERATE THE SUBSAMPLED TRAIN DATA
            (kspace_batch_subsampled, kspace_batch_mask) = self.mask_kspace(
                full_kspace=kspace_batch_full
            )

            (batch_dim, channel_dim, height_dim, width_dim) = kspace_batch_full.shape

            if is_image_space:
                ####
                ####    IMAGE DOMAIN
                ####

                image_fullysampled_label = np.zeros(
                   self.train_images.shape 
                )

                image_subsampled_input = np.zeros(
                   self.train_images.shape 
                )

                for counter in range(batch_dim):
                    image_fullysampled_label[counter, :, :, :] = self.kspace_to_img(
                        kspace_batch_full[counter, 0, :, :]
                    )
                    image_subsampled_input[counter, :, :, :] = self.kspace_to_img(
                        kspace_batch_subsampled[counter, 0, :, :]
                    )

                if return_masks:
                    return (
                        image_subsampled_input,
                        image_fullysampled_label,
                        kspace_batch_mask,
                    )
                else:
                    return image_subsampled_input, image_fullysampled_label

            else:
                ####
                ####    KSPACE DOMAIN
                ####

                # IMAGES ARE SAVED AS

                # kspace_batch_subsampled and kspace_batch_full are of the size
                #   (batch_size,n_channels,img_height,img_width)
                #   (batch,1,height,width)
                #       because they are still complex numbers
                kspace_batch_subsampled_real = np.squeeze(
                    np.real(kspace_batch_subsampled)
                )
                # kspace_batch_subsampled_real = (kspace_batch_subsampled_real-self.kspace_mean[0])/self.kspace_std[1]

                kspace_batch_subsampled_imag = np.squeeze(
                    np.imag(kspace_batch_subsampled)
                )
                # kspace_batch_subsampled_real = (kspace_batch_subsampled_real-self.kspace_mean[1])/self.kspace_std[1]

                kspace_batch_fullysampled_real = np.squeeze(np.real(kspace_batch_full))

                kspace_batch_fullysampled_imag = np.squeeze(np.imag(kspace_batch_full))
                kspace_subsampled_input = np.zeros(
                    (batch_dim, 2, height_dim, width_dim)
                )
                kspace_fullysampled_label = np.zeros(
                    (batch_dim, 2, height_dim, width_dim)
                )

                kspace_subsampled_input[:, 0, :, :] = kspace_batch_subsampled_real
                kspace_subsampled_input[:, 1, :, :] = kspace_batch_subsampled_imag

                kspace_fullysampled_label[:, 0, :, :] = kspace_batch_fullysampled_real
                kspace_fullysampled_label[:, 1, :, :] = kspace_batch_fullysampled_imag

                # print('SHAPES OF OUTPUTS \n\n\n')
                # print(kspace_fullysampled_label.shape)
                # print(kspace_subsampled_input.shape)
                # print(aa)

                tmp = kspace_batch_mask
                kspace_batch_mask = np.zeros((batch_dim, 2, height_dim, width_dim))
                kspace_batch_mask[:, 0, :, :] = np.squeeze(tmp)
                kspace_batch_mask[:, 1, :, :] = np.squeeze(tmp)
                kspace_batch_mask = 1 - kspace_batch_mask

                # kspace_batch_mask = np.logical_not(kspace_batch_mask).astype(np.float32)

                if return_masks:
                    return (
                        kspace_subsampled_input,
                        kspace_fullysampled_label,
                        kspace_batch_mask,
                    )
                else:
                    return kspace_subsampled_input, kspace_fullysampled_label


    def shuffle_data(self):
        # print("\n Shuffling data . . . \n")

        if self.is_2d:
            (total_train, num_channels, img_height, img_width) = self.train_images.shape
            (total_valid, num_channels, img_height, img_width) = self.valid_images.shape

            ########
            ######## RANDOMIZE THE ORDER OF THE TRAIN DATA
            ########

            train_images = self.train_images
            train_kspace = self.train_kspace
            train_names_old = self.train_names

            train_arange = np.arange(total_train)
            np.random.shuffle(train_arange)
            
            train_names = np.array(train_names_old)
            train_names = train_names[train_arange]
            train_names = list(train_names)

            self.train_names = train_names
            self.train_images = train_images[train_arange, :, :, :]
            self.train_kspace = train_kspace[train_arange, :, :, :]

            ########
            ######## RANDOMIZE THE ORDER OF THE VALIDATION DATA
            ########

            valid_images = self.valid_images
            valid_kspace = self.valid_kspace
            valid_names_old = self.valid_names

            valid_arange = np.arange(total_valid)
            np.random.shuffle(valid_arange)

            valid_names = np.array(valid_names_old)
            valid_names = valid_names[valid_arange]
            valid_names = list(valid_names)

            self.valid_names = valid_names
            self.valid_images = valid_images[valid_arange, :, :, :]
            self.valid_kspace = valid_kspace[valid_arange, :, :, :]

        else:
            (total_train, num_channels, img_height, img_width, num_slices) = self.train_images.shape

            ########
            ######## RANDOMIZE THE ORDER OF THE TRAIN DATA
            ########

            train_images = self.train_images
            train_kspace = self.train_kspace
            train_names_old = self.train_names

            train_arange = np.arange(total_train)
            np.random.shuffle(train_arange)
            
            train_names = np.array(train_names_old)
            train_names = train_names[train_arange]
            train_names = list(train_names)

            self.train_names = train_names
            self.train_images = train_images[train_arange, :, :, :, :]
            self.train_kspace = train_kspace[train_arange, :, :, :, :]

            ########
            ######## RANDOMIZE THE ORDER OF THE VALIDATION DATA
            ########

            valid_images = self.valid_images
            valid_kspace = self.valid_kspace
            valid_names_old = self.valid_names

            valid_arange = np.arange(total_valid)
            np.random.shuffle(valid_arange)

            valid_names = np.array(valid_names_old)
            valid_names = valid_names[valid_arange]
            valid_names = list(valid_names)

            self.valid_names = valid_names
            self.valid_images = valid_images[valid_arange, :, :, :, :]
            self.valid_kspace = valid_kspace[valid_arange, :, :, :, :]
            

    def mask_kspace(self, full_kspace):
        num_scans=num_channels=img_height=img_width = 0

        acceleration_factor = self.acceleration_factor
        polynomial_order = self.polynomial_order
        distance_penalty = self.distance_penalty
        center_maintained = self.center_maintained

        if self.is_2d:
            (num_scans, num_channels, img_height, img_width) = full_kspace.shape
        else:
            (num_scans, num_channels, img_height, img_width, num_slices) = full_kspace.shape

        undersampling_factor = 1.0 / acceleration_factor

        undersampled_kspace = np.zeros(full_kspace.shape, dtype=np.cdouble)
        mask_3d = np.zeros((num_scans, num_channels, img_height, img_width), dtype=np.bool)

        for counter in range(num_scans):
            (pdf, offset_value) = gen_pdf(
                # NOTE: This might fail for non square images
                img_size=(1 if self.is_2d else img_height, img_width),
                poly_order=polynomial_order,
                usf=undersampling_factor,
                dist_penalty=distance_penalty,
                radius=center_maintained,
            )
            (sub_mask, sf) = gen_sampling_mask(pdf, max_iter=120, sample_tol=0.5)
            
            mask_2d = sub_mask

            # Take the horiztonal 1D kspace psueorandom mask and replicate it down vertically
            # To create the actual 2D kspace mask that will be used for the data
            #       NOTE: Replicated vertically since we have no penalty in the readout direction
            #       so we don't subsample in readout
            if self.is_2d:
                mask_1d = sub_mask
                mask_2d = np.matlib.repmat(mask_1d, img_height, 1).astype(np.cdouble)

            mask_3d[counter, 0, :, :] = mask_2d

            if self.is_2d:
                undersampled_slice = np.multiply(full_kspace[counter, 0, :, :], mask_2d)
                undersampled_kspace[counter, 0, :, :] = undersampled_slice
            else:
                undersampled_sample = np.multiply(np.reshape(full_kspace[counter, 0, :, :, :], (num_slices, img_height, img_width)), mask_2d)
                undersampled_kspace[counter, 0, :, :, :] = np.reshape(undersampled_sample, (img_height, img_width, num_slices))

        return undersampled_kspace, mask_3d

    def gen_pdf(
        self, img_size, poly_order=3, usf=0.5, dist_penalty=2, radius=0, total_iter=1e4
    ):
        # poly_order -> The order of the polynomial that will be used to
        #       weight the distance from the center of kspace
        #
        # usf -> undersampling factor (was called pctg)
        #
        # dist_penalty -> 1 for L1 norm, 2 for L2 norm
        #
        # radius -> minimum radius of voxel pts around kspace center that must
        #           be sampled
        #           (Not sure if this is an integer or ratio of the ims size?)
        #

        if radius > 1:
            raise ValueError("RADIUS MUST BE BETWEEN 0 AND 1")

        min_offset = 0.0
        max_offset = 1.0

        # NOTE: THESE ARE THE INITIALIZATIONS FOR THE BISECTION METHOD
        #       ROOT FINDING ALGORITHM WE USE

        if len(img_size) < 2:
            img_size = np.array([img_size, 1])

        (img_height, img_width) = img_size
        # Define the image sizes

        total_sample_pts = img_height * img_width
        num_desired_pts = np.floor(img_height * img_width * usf)
        # Undersampling factor times number of points

        if (
            img_size[0] == 1 or img_size[1] == 1
        ):  # A 1D sampling distribution, meaning a 2D image
            r = np.abs(
                np.linspace(
                    start=-1, stop=1, num=max(img_height, img_width), dtype=np.double
                )
            )
            r = r / np.amax(r)  # UNCLEAR IF THIS IS NEEDED?

        else:  # A 2D sampling distribution, meaning a 3D image

            x_lin = np.linspace(start=-1, stop=1, num=img_height, dtype=np.double)
            y_lin = np.linspace(start=-1, stop=1, num=img_width, dtype=np.double)

            [x, y] = np.meshgrid(x_lin, y_lin)  # Create X and Y Meshgrid
            # These correspond to the voxel locations w.r.t the center of kspace
            # -1 is farthest left or up, 1 is fartest right or down

            if dist_penalty == 1:
                r = np.maximum(np.abs(x), np.abs(y))

            elif dist_penalty == 2:
                r = np.sqrt(np.square(x) + np.square(y))
                r = r / np.amax(r)

            else:
                raise ValueError("Distance Penalty is neither L1 or L2 :((((((((")

        indices_to_keep = np.arange(total_sample_pts) + 1
        indices_to_keep = indices_to_keep[r.flatten() < radius] - 1
        # THESE ARE THE LINEAR INDICES OF THE VOXEL POINTS THAT ARE WITHIN THE
        # RADIUS OF THE CENTER OF KSPACE AND MUST BE KEPT

        # plt.imshow(r)
        # plt.show()

        pdf = np.power((1 - r), poly_order).flatten()
        # Since we want the PDF to be higher the closer to the center
        # We take the "r" matrix, which is the distance from the center of kspace
        # and then invert it (so the pdf drops off as you move away from center)
        # Finally, we raise the pdf to the prescribed power

        pdf[indices_to_keep] = 1
        pdf = pdf.reshape((img_height, img_width))
        # This merely sets the values within the prescribed radius from kspace
        # center to 1 to ensure it will be sampled

        if np.floor(pdf.sum()) > num_desired_pts:
            raise ValueError("Infeasible, Try again :((((((")

        # BISECTION METHOD
        #       NOTE: SEARCHES FOR THE VALUE OF "CHECK_POINT" WHICH, WHEN ADDED TO THE PDF
        #           ENSURES THAT THE SUM OF THE ENTIRE MATRIX IS EQUAL TO THE DESIRED NUMBER
        #           OF SAMPLE POINTS

        iter_num = 0

        while iter_num < total_iter:

            check_point = (
                min_offset + max_offset
            ) / 2  # Reset point of interest to avg
            # Between the upper and lower bounds

            pdf = np.power((1 - r), poly_order).flatten() + check_point
            # Create a candidate pdf by taking the weighted distance matrix
            # and adding the checked value to it

            pdf[pdf > 1] = 1  # Truncate to 1
            pdf[indices_to_keep] = 1  # Ensure prescribed center of kspace is sampled
            pdf = pdf.reshape((img_height, img_width))

            num_sampled_pts = np.floor(pdf.sum())

            if num_sampled_pts == num_desired_pts:
                print("PDF CONVERGED ON ITERATION " + str(iter_num) + "\n\n")
                break
            if num_sampled_pts > num_desired_pts:  # Infeasible
                max_offset = check_point
            if num_sampled_pts < num_desired_pts:  # Feasible but not optimal
                min_offset = check_point

            iter_num += 1

        return pdf, check_point

    def gen_sampling_mask(self, pdf, max_iter=150, sample_tol=0.5):
        # pdf -> The simulated pdf from gen_pdf
        #       NOTE: THIS pdf is not normalized, i.e.
        #               integral_-inf^inf != 0
        #   max_iter -> simply the maxiumum iterations
        #
        #   sample_tol -> the acceptable deviation away from the
        #           desired number of sample points

        total_elements = (
            pdf.shape[0] * pdf.shape[1]
        )  # Height times width = total pixels

        img_height, img_width = pdf.shape

        pdf[pdf > 1] = 1  # Truncate at 1

        num_desired_pts = pdf.sum()

        min_transform_interference = 1e40  # Initialize some stupid high value
        # So we only find values lower than this

        min_interference_mask = np.zeros(pdf.shape, dtype=np.bool)
        # Initialize output case it doesnt converge

        for counter in range(max_iter):

            candidate_mask = np.zeros(pdf.shape)
            # Create new candidate sampling mask

            point_difference = np.abs(candidate_mask.sum() - num_desired_pts)
            # The difference between the number of points we want
            # and the number of points in the proposed candidate distribution

            while (
                point_difference > sample_tol
            ):  # Randomly sample points until it meets criteria
                candidate_mask = np.random.uniform(
                    low=0.0, high=1.0, size=total_elements
                )

                candidate_mask = candidate_mask.reshape((img_height, img_width))
                # Uniformly sample points between 0 and 1
                # in a matrix of the same size as the image

                candidate_mask = candidate_mask <= pdf
                # Accept the randomly sampled points that are less than the pdf
                #       NOTE: Since the pdf is set to 1 at certain locations
                #               those will always be sampled. for the values less than 1
                #               the probability of that point is proportional to the pdf val

                point_difference = np.abs(candidate_mask.sum() - num_desired_pts)

            inv_fourier = np.fft.ifft2(np.divide(candidate_mask, pdf))
            # Compute the inverse fourier transform so we can quantify the artifacts
            # that would be created from the candidate distribution

            current_interference = np.amax(np.abs(inv_fourier))
            # Compute the amount of interference (artefacts) in the Fourier??? domain
            # from the candidate distribution

            if current_interference < min_transform_interference:
                # if the proposed distribution has less artefacts than the previous minimum
                # then we save it

                min_transform_interference = current_interference
                min_interference_mask = candidate_mask

        actual_pct_undersampling = min_interference_mask.sum() / total_elements * 100

        # print("\n\n UNDERSAMPLING PCT  " + str(actual_pct_undersampling) + "\n\n")

        return candidate_mask, actual_pct_undersampling

    def img_to_kspace(self, img):
        return fft.ifftshift(fft.fftn(fft.fftshift(img)))

    def kspace_to_img(self, kspace):
        return np.abs(fft.ifftshift(fft.ifftn(fft.fftshift(kspace)))).astype(np.double)


if __name__ == "__main__":
    logger = get_logger('data_loader')
    logger.info('Running data_loader')

    parser = argparse.ArgumentParser(description='Please specify if you would like to use the center 2D slice or whole 3D volume for each scan')
    parser.add_argument('--2d', dest='run_2d', action='store_true')
    parser.add_argument('--3d', dest='run_2d', action='store_false')
    parser.set_defaults(run_2d=True)

    args = parser.parse_args()

    run_2d = args.run_2d

    def img_to_kspace(img):
        return fft.ifftshift(fft.fftn(fft.fftshift(img)))

    def kspace_to_img(kspace):
        return np.abs(fft.ifftshift(fft.ifftn(fft.fftshift(kspace)))).astype(np.double)

    the_generator = data_generator(bool_2d=run_2d)  # Default parameters go in here

    logger.info('Data generator init complete')

    ''' deprecated?
    gen_tf = True

    if gen_tf:
        (
            train_images,
            train_kspace,
            train_names,
            train_kspace_undersampled,
            valid_images,
            valid_kspace,
            valid_names,
            valid_kspace_undersampled,
            # img_mean,
            # img_std
        ) = the_generator.get_batch_tf()

        print(train_images.shape)
        print(train_kspace.shape)
        print(train_kspace_undersampled.shape)
        logger.info('Completed generating tensors')
        logger.info('Shape of train_images: ' + str(train_images.shape))

    else:
        (
            train_images,
            train_kspace,
            train_names,
            train_kspace_undersampled,
            train_kspace_mask,
            valid_images,
            valid_kspace,
            valid_names,
            valid_kspace_undersampled,
            valid_kspace_mask,
            # img_mean,
            # img_std
        ) = the_generator.get_batch()

        # print('SHAPES')
        # print(train_images.shape)
        # print(train_kspace.shape)
        # print(valid_images.shape)
        # print(valid_kspace.shape)

        train_images = np.transpose(np.squeeze(train_images), (1, 2, 0))
        train_kspace = np.transpose(np.squeeze(train_kspace), (1, 2, 0))
        train_kspace_undersampled = np.transpose(
            np.squeeze(train_kspace_undersampled), (1, 2, 0)
        )

        # print('SHAPES')
        # print(train_images.shape)
        # print(train_kspace.shape)

        (img_height, img_width, num_scans) = train_images.shape

        train_downsampled_img = np.zeros(train_images.shape)

        for cc in range(num_scans):
            train_downsampled_img[:, :, cc] = kspace_to_img(
                train_kspace_undersampled[:, :, cc]
            )

        full_kspace_mask = np.ones((train_images.shape))

        plot_train_images = np.hstack((train_images, train_downsampled_img))
        plot_train_kspace = np.hstack((train_kspace, train_kspace_undersampled))

        show_3d_images(train_downsampled_img)

        # show_3d_images(plot_train_kspace)
        show_3d_images(plot_train_images)
    '''
