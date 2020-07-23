###############################
###############################
###############################
###############################
#########
#########
#########   CREATED BY: BRANDON CLINTON JONES
#########       APRIL 22, 2020
#########
#########
#########
###############################
###############################
###############################
###############################


import tensorflow as tf
import numpy as np
import timeit
from tensorflow.python import roll
import os
from matplotlib import pyplot as plt
import numpy.fft as fft


#############
#############
#############       CUSTOM DEPENDENCIES!!!!!
#############
#############

import model_architectures
from data_loader_class import data_generator
from show_3d_images import show_3d_images

# from main_BCJ import CNN
from main_BCJ import CNN

if __name__ == "__main__":

    def img_to_kspace(img):
        return fft.ifftshift(fft.fftn(fft.fftshift(img)))

    def kspace_to_img(kspace):
        return np.abs(fft.ifftshift(fft.ifftn(fft.fftshift(kspace)))).astype(np.double)

    batch_size = 1
    acc_factor = 5
    max_epoch = 500
    polyfit = 4
    lr = 2e-6

    # name = 'test_kspace_loss'

    name = "b_{}_af_{}_e_{}_pf_{}_lr_{}_imagekspace_justdy".format(
        str(batch_size), str(acc_factor), str(max_epoch), str(polyfit), str(lr)
    )

    if os.path.isdir("E:\\ENM_Project\\SPGR_AF2_242_242_1500_um\\"):
        study_dir = "E:\\ENM_Project\\SPGR_AF2_242_242_1500_um\\"
        project_folder = "E:\\ENM_Project\\"
        save_dir = project_folder + "Saved_Models\\ImageSpace\\" + name + "\\"
        print("WINDOWS\n\n")
    elif os.path.isdir("/run/media/bellowz"):
        study_dir = "/run/media/bellowz/S*/ENM_Project/SPGR_AF2_242_242_1500_um/"
        project_folder = "/run/media/bellowz/Seagate Backup Plus Drive/ENM_Project/"
        save_dir = project_folder + "Saved_Models/ImageSpace/" + name + "/"
        print("LINUX \n\n")

    convnet = CNN(
        project_folder=project_folder,
        batch_size=batch_size,
        max_epoch=100,
        model_name=name,
        learn_rate=lr,
        is_image_space=True,
        acceleration_factor=acc_factor,
        polyfit_order=polyfit,
    )

    print("LOADING NETWORK . . . \n")

    convnet.load()

    print("LOADED\n\n")

    plot_acceleration_factors = np.array([3, 5, 7])

    for current_af in plot_acceleration_factors:

        save_dir_current_af = save_dir + "Plots_AF" + str(current_af) + "/"

        if not os.path.isdir(save_dir_current_af):
            os.makedirs(save_dir_current_af)

        print("LOADING SCAN DATA . . . \n\n")

        my_gen = data_generator(
            batch_size=batch_size,
            acceleration_factor=current_af,
            poly_distance_order=current_af,
            bool_shuffle=False,
        )

        print("FINISHED LOADING SCAN DATA . . . \n\n")

        (
            num_train,
            num_valid,
            num_channels,
            img_height,
            img_width,
            train_names,
            valid_names,
        ) = my_gen.get_info()

        steps_per_epoch_train = int(np.ceil(num_train / batch_size))

        # image_labels_train = np.zeros((num_train,img_height,img_width))
        # image_downsampled_train = np.zeros((num_train,img_height,img_width))
        # image_predicted_train = np.zeros((num_train,img_height,img_width))

        is_image_space = True

        # print('PREDICTING TRAIN SET . . . \n\n')
        # for counter in range(steps_per_epoch_train):
        # 	# print('\n\n PREDICTING BATCH  '+str(counter))

        # 	batch_input_subsampled_train, batch_label_fullysampled_train = my_gen.generator(batch_ind=counter,
        # 	is_train=True,
        # 	is_image_space=is_image_space,
        # 	return_masks = False)

        # 	batch_output_cnn_interpolated = convnet.predict( batch_input_subsampled_train)

        # 	start_ind = counter * batch_size
        # 	end_ind = start_ind + batch_size

        # 	if end_ind > num_train:
        # 		end_ind = num_train
        # 		start_ind = num_train-batch_size

        # 	image_labels_train[start_ind:end_ind,:,:]=np.squeeze(batch_label_fullysampled_train)
        # 	image_downsampled_train[start_ind:end_ind,:,:]=np.squeeze(batch_input_subsampled_train)
        # 	image_predicted_train[start_ind:end_ind,:,:]=np.squeeze(batch_output_cnn_interpolated)

        # image_labels_train = np.transpose(image_labels_train,(1,2,0))
        # image_downsampled_train = np.transpose(image_downsampled_train,(1,2,0))
        # image_predicted_train = np.transpose(image_predicted_train,(1,2,0))

        # plots_train = np.hstack((image_labels_train,image_downsampled_train,image_predicted_train))
        # # show_3d_images(plots)

        # for cc in range(num_train):
        # 	tmp_img = plots_train[:,:,cc]
        # 	tmp_str = save_dir_current_af+'TRAIN_00'+str(cc)+'.png'

        # 	plt.imsave(tmp_str,tmp_img,cmap='gray')

        steps_per_epoch_valid = int(num_valid / batch_size)

        image_labels_valid = np.zeros((num_valid, img_height, img_width))
        image_downsampled_valid = np.zeros((num_valid, img_height, img_width))
        image_predicted_valid = np.zeros((num_valid, img_height, img_width))

        print("PREDICTING VALIDATION SET . . . \n\n\n")

        for counter in range(steps_per_epoch_valid):

            batch_input_subsampled_valid, batch_label_fullysampled_valid = my_gen.generator(
                batch_ind=counter,
                is_train=False,
                is_image_space=is_image_space,
                return_masks=False,
            )

            batch_output_cnn_interpolated = convnet.predict(
                batch_input_subsampled_valid
            )

            start_ind = counter * batch_size
            end_ind = start_ind + batch_size

            if end_ind > num_valid - 1:
                end_ind = num_valid
                start_ind = num_valid - batch_size

            image_labels_valid[start_ind:end_ind, :, :] = np.squeeze(
                batch_label_fullysampled_valid
            )
            image_downsampled_valid[start_ind:end_ind, :, :] = np.squeeze(
                batch_input_subsampled_valid
            )
            image_predicted_valid[start_ind:end_ind, :, :] = np.squeeze(
                batch_output_cnn_interpolated
            )

        image_labels_valid = np.transpose(image_labels_valid, (1, 2, 0))
        image_downsampled_valid = np.transpose(image_downsampled_valid, (1, 2, 0))
        image_predicted_valid = np.transpose(image_predicted_valid, (1, 2, 0))

        plots_valid = np.hstack(
            (image_labels_valid, image_downsampled_valid, image_predicted_valid)
        )

        # for cc in range(num_valid):
        # 	tmp_img = plots_valid[:,:,cc]
        # 	tmp_str = save_dir_current_af+'VALID_00'+str(cc)+'.png'

        # 	plt.imsave(tmp_str,tmp_img,cmap='gray')

        tmp_img = plots_valid[:, :, 14]
        # tmp_str = save_dir_current_af+'VALID_00'+str(cc)+'.png'

        # plt.imshow(tmp_img, cmap="gray")
        # plt.show()

        plot_downsampled = image_downsampled_valid[:, :, 14]
        plot_label = image_labels_valid[:, :, 14]
        plot_prediction = image_predicted_valid[:, :, 14]

        label_kspace = img_to_kspace(plot_label)

        plot_aliased = np.zeros(plot_label.shape, dtype=label_kspace.dtype)
        center_kspace = np.zeros(plot_label.shape, dtype=label_kspace.dtype)

        img_height, img_width = plot_label.shape

        sampling_pts = np.arange(0, img_width, current_af)
        mask_1d = np.zeros((1, img_width))
        mask_1d[:, sampling_pts] = 1
        mask_2d = np.matlib.repmat(mask_1d, img_height, 1)  # .astype(np.cdouble)

        plot_aliased = np.multiply(label_kspace, mask_2d)
        plot_aliased = kspace_to_img(plot_aliased)

        center_kspace_location = int(img_width / 2)
        sampling_radius = int(img_width / 2 / current_af)
        center_kspace[
            :,
            (center_kspace_location - sampling_radius) : (
                center_kspace_location + sampling_radius
            ),
        ] = 1

        # plt.imshow(center_kspace.astype(np.uint8),cmap='gray')
        # plt.show()

        center_kspace = np.multiply(label_kspace, center_kspace)
        center_kspace = kspace_to_img(center_kspace)

        abstract_dir = "E:\\ENM_Project\\"

        plt.imsave(
            abstract_dir + "MC_downsampled_" + str(current_af) + ".png",
            plot_downsampled,
            cmap="gray",
        )
        plt.imsave(abstract_dir + "Label.png", plot_label, cmap="gray")
        plt.imsave(
            abstract_dir + "Predicted_" + str(current_af) + ".png",
            plot_prediction,
            cmap="gray",
        )
        plt.imsave(
            abstract_dir + "aliased_" + str(current_af) + ".png",
            plot_aliased,
            cmap="gray",
        )
        plt.imsave(
            abstract_dir + "centerdownsampled_" + str(current_af) + ".png",
            center_kspace,
            cmap="gray",
        )

        residual_predicted = np.abs(plot_label - plot_prediction)
        residual_mc_downsampled = np.abs(plot_label - plot_downsampled)

        # plt.imshow(residual_predicted)
        # plt.show()

        # plt.imshow(residual_mc_downsampled)
        # plt.show()

        # plt.hist(residual_predicted)
        # plt.show()

        # plt.hist(residual_mc_downsampled)
        # plt.show()

        print("MAX MIN RESIDUALS")
        print(np.amax(residual_mc_downsampled))
        print(np.amin(residual_mc_downsampled))
        print(np.amax(residual_predicted))
        print(np.amin(residual_predicted))

        tmp = np.hstack((residual_predicted, residual_mc_downsampled))

        # max_residual = np.amax(tmp)
        max_residual = 0.3

        # plt.imshow(tmp,cmap="gray",vmax=max_residual)
        # plt.show()

        plt.imsave(
            abstract_dir + "residual_predicted_" + str(current_af) + ".png",
            residual_predicted,
            cmap="gray",
            vmax=max_residual,
        )
        plt.imsave(
            abstract_dir + "residual_mc_downsampled_" + str(current_af) + ".png",
            residual_mc_downsampled,
            cmap="gray",
            vmax=max_residual,
        )

        # plt.imsave(save_dir_current_af+'pendergrass.png',tmp_img,cmap='gray')

        # show_3d_images(plots)

