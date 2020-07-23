###############################
###############################
###############################
###############################
#########
#########
#########   CREATED BY: BRANDON CLINTON JONES
#########           AND CARLOS ADOLFO OSUNA
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
import numpy.fft as fft
import pandas as pd

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
from evals_2 import my_metrics_2D


def img_to_kspace(img):
    return fft.ifftshift(fft.fftn(fft.fftshift(img)))


def kspace_to_img(kspace):
    return np.abs(fft.ifftshift(fft.ifftn(fft.fftshift(kspace)))).astype(np.double)


def kspace_output_to_image_stack(kspace_stack):
    # The kspace is input size
    # (Batch_dim, real/imag, height, width)

    (batch_dim, real_imag_dim, img_height, img_width) = kspace_stack.shape

    output_image = np.zeros((batch_dim, img_height, img_width))

    for counter in range(batch_dim):
        tmp_real = np.squeeze(kspace_stack[counter, 0, :, :])
        tmp_imag = np.squeeze(kspace_stack[counter, 1, :, :])

        tmp_kspace = tmp_real + 1j * tmp_imag
        tmp_image = kspace_to_img(tmp_kspace)
        output_image[counter, :, :] = tmp_image

    return output_image



if __name__ == "__main__":

    if os.path.isdir("E:\\ENM_Project\\SPGR_AF2_242_242_1500_um\\"):
        study_dir = "E:\\ENM_Project\\SPGR_AF2_242_242_1500_um\\"
        project_folder = "E:\\ENM_Project\\"
        print("WINDOWS\n\n")
    elif os.path.isdir("/run/media/bellowz"):
        study_dir = "/run/media/bellowz/S*/ENM_Project/SPGR_AF2_242_242_1500_um/"
        project_folder = "/run/media/bellowz/Seagate Backup Plus Drive/ENM_Project/"
        print("LINUX \n\n")

    af_eval = 7
    pf_eval = 6

    is_image_space = True
    batch_size = 1
    acc_factor = 5
    max_epoch = 500
    polyfit = 4
    lr = 2e-6

    # name = 'test_kspace_loss'

    name = "b_{}_af_{}_e_{}_pf_{}_lr_{}_imagekspace_justdy".format(
        str(batch_size), str(acc_factor), str(max_epoch), str(polyfit), str(lr)
    )
    save_dir = (
        "/run/media/bellowz/Seagate Backup Plus Drive/ENM_Project/abstract_stuff/"
    )

    convnet = CNN(
        project_folder=project_folder,
        batch_size=batch_size,
        max_epoch=100,
        model_name=name,
        learn_rate=lr,
        is_image_space=is_image_space,
        acceleration_factor=acc_factor,
        polyfit_order=polyfit,
    )

    # is_image_space = False
    # batch_size = 3
    # acc_factor = 5
    # max_epoch = 200
    # polyfit = 4
    # lr = 1e-4
    # # name = 'test_kspace_loss'

    # name = "b_{}_af_{}_e_{}_pf_{}_lr_{}".format(
    #     str(batch_size), str(acc_factor), str(max_epoch), str(polyfit), str(lr)
    # )

    # convnet = CNN(
    #     project_folder=project_folder,
    #     batch_size=batch_size,
    #     max_epoch=max_epoch,
    #     model_name=name,
    #     learn_rate=lr,
    #     is_image_space=is_image_space,
    #     acceleration_factor=acc_factor,
    #     polyfit_order=polyfit,
    # )

    print("LOADING NETWORK . . . \n")

    convnet.load()

    print("LOADED\n\n")

    print("LOADING SCAN DATA . . . \n\n")

    my_gen = data_generator(
        batch_size=batch_size,
        acceleration_factor=af_eval,
        poly_distance_order=pf_eval,
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

    image_labels = np.zeros((num_train, img_height, img_width))
    image_downsampled = np.zeros((num_train, img_height, img_width))
    image_predicted = np.zeros((num_train, img_height, img_width))

    print("PREDICTING TRAIN SET . . . \n\n")
    for counter in range(steps_per_epoch_train):
        # print('\n\n PREDICTING BATCH  '+str(counter))

        batch_input_subsampled_train, batch_label_fullysampled_train = my_gen.generator(
            batch_ind=counter,
            is_train=True,
            is_image_space=is_image_space,
            return_masks=False,
        )

        batch_output_cnn_interpolated = convnet.predict(batch_input_subsampled_train)

        if not is_image_space:
            batch_output_cnn_interpolated = kspace_output_to_image_stack(
                batch_output_cnn_interpolated
            )
            batch_label_fullysampled_train = kspace_output_to_image_stack(
                batch_label_fullysampled_train
            )
            batch_input_subsampled_train = kspace_output_to_image_stack(
                batch_input_subsampled_train
            )

        start_ind = counter * batch_size
        end_ind = start_ind + batch_size

        if end_ind > num_train:
            end_ind = num_train
            start_ind = num_train - batch_size

        image_labels[start_ind:end_ind, :, :] = np.squeeze(
            batch_label_fullysampled_train
        )
        image_downsampled[start_ind:end_ind, :, :] = np.squeeze(
            batch_input_subsampled_train
        )
        image_predicted[start_ind:end_ind, :, :] = np.squeeze(
            batch_output_cnn_interpolated
        )

    image_labels = np.transpose(image_labels, (1, 2, 0)).clip(0, 1)
    image_downsampled = np.transpose(image_downsampled, (1, 2, 0)).clip(0, 1)
    image_predicted = np.transpose(image_predicted, (1, 2, 0)).clip(0, 1)

    train_data_frame = my_metrics_2D(
        label=image_labels, subsampled=image_downsampled, prediction=image_predicted
    )

    headers = [
        "DOWNSAMPLED SSIM",
        "DOWNSAMPLED NRMSE",
        "DOWNSAMPLED PSNR",
        "PREDICTED SSIM",
        "PREDICTED NRMSE",
        "PREDICTED PSNR",
    ]

    df_train = pd.DataFrame(data=train_data_frame, columns=headers, index=train_names)

    train_filname = save_dir + "train_AF" + str(af_eval) + "_results.csv"
    df_train.to_csv(train_filname)

    # plots = np.hstack((image_labels, image_downsampled, image_predicted))
    # show_3d_images(plots)

    # ssim_downsampled_mean = np.mean(ssim_downsampled)
    # ssim_downsampled_std = np.std(ssim_downsampled)

    # nrmse_downsampled_mean = np.mean(nrmse_downsampled)
    # nrmse_downsampled_std = np.std(nrmse_downsampled)

    # psnr_downsampled_mean = np.mean(psnr_downsampled)
    # psnr_downsampled_std = np.std(psnr_downsampled)


    # print('DOWNSAMPLED METRICS . . . \n\n')
    # print('SSIM')
    # print(ssim_downsampled_mean)
    # print(ssim_downsampled_std)
    # print('NRMSE')
    # print(nrmse_downsampled_mean)
    # print(nrmse_downsampled_std)
    # print('PSNR')
    # print(psnr_downsampled_mean)
    # print(psnr_downsampled_std)

    # ssim_predicted_mean = np.mean(ssim_predicted)
    # ssim_predicted_std = np.std(ssim_predicted)

    # nrmse_predicted_mean = np.mean(nrmse_predicted)
    # nrmse_predicted_std = np.std(nrmse_predicted)

    # psnr_predicted_mean = np.mean(psnr_predicted)
    # psnr_predicted_std = np.std(psnr_predicted)


    # print('PREDICTED METRICS . . . \n\n')
    # print('SSIM')
    # print(ssim_predicted_mean)
    # print(ssim_predicted_std)
    # print('NRMSE')
    # print(nrmse_predicted_mean)
    # print(nrmse_predicted_std)
    # print('PSNR')
    # print(psnr_predicted_mean)
    # print(psnr_predicted_std)


    steps_per_epoch_valid = int(num_valid / batch_size)

    image_labels = np.zeros((num_valid, img_height, img_width))
    image_downsampled = np.zeros((num_valid, img_height, img_width))
    image_predicted = np.zeros((num_valid, img_height, img_width))

    print("PREDICTING VALIDATION SET . . . \n\n\n")

    for counter in range(steps_per_epoch_valid):

        batch_input_subsampled_valid, batch_label_fullysampled_valid = my_gen.generator(
            batch_ind=counter,
            is_train=False,
            is_image_space=is_image_space,
            return_masks=False,
        )

        batch_output_cnn_interpolated = convnet.predict(batch_input_subsampled_valid)

        if not is_image_space:
            batch_output_cnn_interpolated = kspace_output_to_image_stack(
                batch_output_cnn_interpolated
            )
            batch_label_fullysampled_valid = kspace_output_to_image_stack(
                batch_label_fullysampled_valid
            )
            batch_input_subsampled_valid = kspace_output_to_image_stack(
                batch_input_subsampled_valid
            )

        start_ind = counter * batch_size
        end_ind = start_ind + batch_size

        if end_ind > num_valid - 1:
            end_ind = num_valid
            start_ind = num_valid - batch_size

        image_labels[start_ind:end_ind, :, :] = np.squeeze(
            batch_label_fullysampled_valid
        )
        image_downsampled[start_ind:end_ind, :, :] = np.squeeze(
            batch_input_subsampled_valid
        )
        image_predicted[start_ind:end_ind, :, :] = np.squeeze(
            batch_output_cnn_interpolated
        )

    image_labels = np.transpose(image_labels, (1, 2, 0)).clip(0, 1)
    image_downsampled = np.transpose(image_downsampled, (1, 2, 0)).clip(0, 1)
    image_predicted = np.transpose(image_predicted, (1, 2, 0)).clip(0, 1)

    valid_data_frame = my_metrics_2D(
        label=image_labels, subsampled=image_downsampled, prediction=image_predicted
    )

    df_train = pd.DataFrame(data=valid_data_frame, columns=headers, index=valid_names)

    valid_filname = save_dir + "valid_AF" + str(af_eval) + "_results.csv"
    df_train.to_csv(valid_filname)

    plots = np.hstack((image_labels, image_downsampled, image_predicted))
    show_3d_images(plots)
