###############################
###############################
###############################
###############################
#########
#########
#########   CREATED BY: BRANDON CLINTON JONES
#########		AND CARLOS ADOLFO OSUNA
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
from tensorflow import fft2d
from tensorflow import ifft2d
import timeit
from tensorflow.python import roll
import tqdm
import os
import glob
from tensorflow.keras.layers import Lambda


#############
#############
#############       CUSTOM DEPENDENCIES!!!!!
#############
#############

import model_architectures

# from model_architectures import unet_7_layers, simple_cnn
from data_loader_class import data_generator
from show_3d_images import show_3d_images


class CNN:
    # class CNN(tf.Module):
    def __init__(
        self,
        bool_2d=True, 
        #project_folder,
        batch_size,
        max_epoch,
        model_name,
        learn_rate=1e-3,
        acceleration_factor=4,
        polyfit_order=3,
        *layerinfo
    ):
        """
            Defines the CNN structure

            Parameters:
                layerinfo = list of tuples containing a tf.layers function
                    and its parameters in kwargs format
                input_shape = expected image shape for inputs
                # NOTE: input_shape is for a given batch
                # 
                # input_shape = ( num_channels, img_height, img_width, num_slices )

            Outputs:
                image = CNN-processed image 
            """

        tf.logging.set_verbosity(tf.logging.ERROR)
        tf.set_random_seed(seed=1)

        #self.project_folder = project_folder

        self.layerinfo = layerinfo
        self.training_loss = []
        self.valid_loss = []

        self.learn_rate = learn_rate

        self.max_epoch = int(max_epoch)

        self.batch_size = batch_size
        self.num_channels = 1
        self.img_height = 512
        self.img_width = 512

        self.acceleration_factor = acceleration_factor
        self.dtype = tf.float32

        self.batch_size = batch_size
        self.model_name = model_name

        self.bool_2d = bool_2d

        self.my_gen = data_generator(
            batch_size=self.batch_size,
            acceleration_factor=self.acceleration_factor,
            poly_distance_order=polyfit_order,
            bool_2d=self.bool_2d
        )

        (
            self.num_train,
            self.num_valid,
            self.num_channels,
            self.img_height,
            self.img_width,
            _,
            _,
        ) = self.my_gen.get_info()

        real_imag_dim = 2

        config_options = tf.ConfigProto()
        # config_options = tf.ConfigProto(log_device_placement=True)
        # Log Device Placement prints which operations are performed
        config_options.gpu_options.allow_growth = True
        # Allow the GPU to use its maximum resources
        self.sess = tf.Session(config=config_options)
        # Define the Tensorflow Session

        # ####
        # #### IMAGE SPACE - 4 DIMENSIONS:
        # ####
        # Batch_dim is set to None so it can be modified later on
        # Num_channels should be size 1

        self.input_matrix_shape = (
            None,
            self.num_channels,
            self.img_height,
            self.img_width,
        )

        self.input_subsampled_placeholder = tf.placeholder(
            dtype=self.dtype, shape=self.input_matrix_shape
        )

        self.label_fullysampled_placeholder = tf.placeholder(
            dtype=self.dtype, shape=self.input_matrix_shape
        )

        self.output_predicted_placeholder = self.forward_pass(
            tensor_input=self.input_subsampled_placeholder
        )

        self.kspace_mask_placeholder = tf.placeholder(
            dtype=self.dtype, shape=self.input_matrix_shape
        )

        # self.loss = tf.losses.mean_squared_error(
        #     labels=self.label_fullysampled_placeholder,
        #     predictions=self.output_predicted_placeholder,
        # )

        self.loss = self.custom_image_loss(
            y_true=self.label_fullysampled_placeholder,
            y_pred=self.output_predicted_placeholder,
            kspace_mask=self.kspace_mask_placeholder,
        )

        self.optimizer_type = tf.train.AdamOptimizer(learning_rate=self.learn_rate)

        self.train_optimizer = self.optimizer_type.minimize(self.loss)

        init = tf.global_variables_initializer()

        self.sess.run(init)

        self.saver = tf.train.Saver()

    def forward_pass(self, tensor_input):
        """
            Runs the image through the layer structure.

            """

        # if issubclass(type(input), tf.Tensor):
        #     # If the input is not a tensor, then we switch from Numpy to a Tensor object
        #     # And define boolean so we reverse the operation if needed
        #     casted = True
        #     original_class = type(input)
        #     # input = tf.Tensor(input)
        #     input_tf = tf.convert_to_tensor(input, dtype=tf.complex128)

        # else:
        #     casted = False


        tensor_output = model_architectures.unet_9_layers(tensor_input)

        return tensor_output

    def train(self):
        """
            Call syntax:
                model.train(inputs,expectations,epochs = 1e3)
                NOTE: inputs is subsampled kspace
                NOTE: expectations is fully sampled kspace or image?
            
            Takes in the following training parameter arguments:
                epochs:int = # of iterations to optimize the loss function
                batchsize:int = size of subset of data to use for an optimizing epoch
                        - Default: None [no batching]

            Can take in the following *args:
                None
            
            Can take in the following **kwargs:
                loss = loss function to be optimized which takes in the following arguments:
                        - loss(self,inputs,expectations,*lossargs,**losskwargs)
                        where lossargs and losskwargs can be defined as additional **kwargs.
                        - Default: tf.nn.l2_loss [decorated to accept inputs in expected format]
                optimizer = optimizer function to perform optimization on loss function, taking on:
                        - optimizer(self.parameters(),*optimizerargs,**optimizerkwargs)
                        where optimizerargs and optimizerkwargs can be defined as additional **kwargs.
                        - Default: tf.compat.v1.train.AdamOptimizer()
                cutoff:callable = cutoff function to truncate optimization, taking:
                        - cutoff(self,lossoutputs,iter,*cutoffargs,**cutoffkwargs)
                        where lossoutputs is a numpy.ndarray of the history of loss evaluations
                        and iter is the iteration #
                """

        ##############
        ##############
        ############## REPLACE THIS WITH DATA LOADER HERE
        ##############
        ##############

        # inputs, expectations, kwargs = updatedefaults(inputs, expectations, kwargs)

        # NOTE: THE CODE BELOW IS THE ACTUAL TRAINING LOOP ITERATION
        # Will need to be modified to work for tensorflow

        # Define the image generator
        # NOTE: MOve this to the init once its done
        # TODO: save the model when you're confident lol
        # save_str = self.save_dir + self.model_name

        steps_per_epoch_train = int(np.ceil(self.num_train / self.batch_size))
        steps_per_epoch_valid = int(np.ceil(self.num_valid / self.batch_size))

        train_epoch_loss = []
        valid_epoch_loss = []

        start_time = timeit.default_timer()

        for epoch_num in range(self.max_epoch):

            print("\n\n EPOCH NUMBER " + str(epoch_num + 1))

            train_batch_loss = []
            valid_batch_loss = []

            # for counter in range(steps_per_epoch_train):
            for counter in tqdm.tqdm(range(steps_per_epoch_train)):

                batch_input_subsampled_train, batch_label_fullysampled_train, batch_kspace_mask_train = self.my_gen.generator(
                    batch_ind=counter, is_train=True
                )


                tf_dict_train = {
                    self.input_subsampled_placeholder: batch_input_subsampled_train,
                    self.label_fullysampled_placeholder: batch_label_fullysampled_train,
                    self.kspace_mask_placeholder: batch_kspace_mask_train,
                }


                # Run a forward pass and backpropagation and output the optimizer state and loss value
                _, training_loss_value = self.sess.run(
                    [self.train_optimizer, self.loss], tf_dict_train
                )

                train_batch_loss.append(training_loss_value)

            train_batch_loss = np.asarray(train_batch_loss).mean()

            elapsed = timeit.default_timer() - start_time

            print(
                "TRAIN ==> Epoch [%d/%d], Loss: %.12f, Time: %2fs"
                % (epoch_num + 1, self.max_epoch, training_loss_value, elapsed)
            )
            start_time = timeit.default_timer()

            train_epoch_loss.append(train_batch_loss)

            # for counter in range(steps_per_epoch_valid):
            for counter in tqdm.tqdm(range(steps_per_epoch_valid)):

                (
                    batch_input_subsampled_valid,
                    batch_label_fullysampled_valid,
                    batch_kspace_mask_valid,
                ) = self.my_gen.generator(
                    batch_ind=counter,
                    is_train=False,
                )


                tf_dict_valid = {
                    self.input_subsampled_placeholder: batch_input_subsampled_valid,
                    self.label_fullysampled_placeholder: batch_label_fullysampled_valid,
                    self.kspace_mask_placeholder: batch_kspace_mask_valid,
                }

                # Run a forward pass without backpropagation and save loss value
                valid_batch_loss = self.sess.run(self.loss, tf_dict_valid)

                valid_epoch_loss.append(valid_batch_loss)

            valid_batch_loss = np.asarray(valid_batch_loss).mean()

            elapsed = timeit.default_timer() - start_time

            print(
                "VALID ==> Epoch [%d/%d], Loss: %.12f, Time: %2fs"
                % (epoch_num + 1, self.max_epoch, valid_batch_loss, elapsed)
            )
            start_time = timeit.default_timer()

            valid_epoch_loss.append(valid_batch_loss)

            if (epoch_num + 1) % 100 == 0:
                print("SAVING MODEL . . . ")
                # TODO: Save model in future
                #self.saver.save(self.sess, save_str, global_step=epoch_num + 1)
        # TODO: save the model
        #self.saver.save(self.sess, save_str, global_step=epoch_num + 1)

    def load(self):

        tf.reset_default_graph()

        meta_graph_name = self.save_dir + self.model_name + "*.meta"

        files_in_dir = glob.glob(meta_graph_name)

        num_files = len(files_in_dir)

        meta_graph_name = files_in_dir[num_files - 1]
        # Grabs the last saved checkpoint ih the directory. Assuming last one is
        # the most trained one

        self.save_dir = os.path.dirname(meta_graph_name)

        self.save_model_name = meta_graph_name[0 : len(meta_graph_name) - 5]

        self.saver = tf.train.import_meta_graph(meta_graph_name)

        self.saver.restore(self.sess, self.save_model_name)

    def load_submission(self, model_location):

        tf.reset_default_graph()

        meta_graph_name = model_location + "*.meta"

        files_in_dir = glob.glob(meta_graph_name)

        num_files = len(files_in_dir)

        meta_graph_name = files_in_dir[num_files - 1]
        # Grabs the last saved checkpoint ih the directory. Assuming last one is
        # the most trained one

        self.save_dir = os.path.dirname(meta_graph_name)

        self.save_model_name = meta_graph_name[0 : len(meta_graph_name) - 5]

        self.saver = tf.train.import_meta_graph(meta_graph_name)

        self.saver.restore(self.sess, self.save_model_name)

    def predict(self, X_star):

        predict_dictionary = {self.input_subsampled_placeholder: X_star}

        predicted_label = self.sess.run(
            self.output_predicted_placeholder, predict_dictionary
        )

        return predicted_label


    def fftshift(self, x):
        """
        Shift the zero-frequency component to the center of the spectrum.
        This function swaps half-spaces for all axes listed (defaults to all).
        Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.
        Parameters
        ----------
        x : array_like, Tensor
            Input array.
        axes : int or shape tuple, optional
            Axes over which to shift.  Default is None, which shifts all axes.
        Returns
        -------
        y : Tensor.
        """
        x_dim = self.img_height
        y_dim = self.img_width
        shift = [int(x_dim // 2), int(y_dim // 2)]

        # print('\n\n\n')
        # print(x.shape)
        # print('\n\n\n')

        return roll(input=x, shift=shift, axis=[2, 3])

    def ifftshift(self, x):
        """
        The inverse of `fftshift`. Although identical for even-length `x`, the
        functions differ by one sample for odd-length `x`.
        Parameters
        ----------
        x : array_like, Tensor.
        axes : int or shape tuple, optional
            Axes over which to calculate.  Defaults to None, which shifts all axes.
        Returns
        -------
        y : Tensor.
        """
        ### SIZE ==> (BATCH,CHANNELS,HEIGHT,WIDTH)

        x_dim = self.img_height
        y_dim = self.img_width

        # print('\n\n\n')
        # print(x.shape)
        # print('\n\n\n')

        shift = [-int(x_dim // 2), -int(y_dim // 2)]
        return roll(input=x, shift=shift, axis=[2, 3])

    def kspace_to_image(self, kspace):

        # NOTE: SO THE INPUT KSPACE MATRIX IS OF SIZE
        # (BATCH_DIM,NUM_CHANNELS,IMG_HEIGHT,IMG_WIDTH)
        # OR   (BATCH,2,HEIGHT,WIDTH)   WHERE 1ST CHANNEL IS REAL
        # 2ND CHANNEL IS IMAGE
        # NEED TO CONVERT TO COMPLEX TENSOR

        # kspace_real = kspace[:,0,:,:]
        # kspace_imag = kspace[:,1,:,:]

        kspace_real = Lambda(lambda y_true: kspace[:, 0, :, :])(kspace)
        kspace_imag = Lambda(lambda y_true: kspace[:, 1, :, :])(kspace)

        image = tf.complex(real=kspace_real, imag=kspace_imag)

        image = self.fftshift(image)
        image = ifft2d(image)
        image = self.ifftshift(image)
        image = tf.math.abs(image)

        return image

    def image_to_kspace(self, image):

        kspace = tf.to_complex64(image)
        kspace = self.ifftshift(kspace)
        kspace = fft2d(kspace)
        kspace = self.fftshift(kspace)
        return kspace

    def custom_kspace_loss(self, y_true, y_pred, not_kspace_mask):
        # Images are of size (batch_dim, n_channels, height, width)
        #
        #   Kspace mask is of size (batch_dim, n_channels, height, width )
        #

        # y_true_real = Lambda( lambda y_true: y_true[:,0,:,:] )(y_true)
        # y_true_imag = Lambda( lambda y_true: y_true[:,1,:,:] )(y_true)

        # y_pred_real = Lambda( lambda y_pred: y_pred[:,0,:,:] )(y_pred)
        # y_pred_imag = Lambda( lambda y_pred: y_pred[:,1,:,:] )(y_pred)

        # kspace_loss_real = tf.losses.mean_squared_error(labels=y_true_real, predictions=y_pred_real)
        # kspace_loss_imag = tf.losses.mean_squared_error(labels=y_true_imag, predictions=y_pred_imag)

        kspace_loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

        image_true = self.kspace_to_image(kspace=y_true)
        image_pred = self.kspace_to_image(kspace=y_pred)

        image_loss = (
            tf.losses.mean_squared_error(labels=image_true, predictions=image_pred)
            * 2e4
        )

        # loss = image_loss
        # loss = kspace_loss
        loss = kspace_loss + image_loss

        return loss


    def custom_image_loss(self, y_true, y_pred, kspace_mask):

        # Images are of size (batch_dim, n_channels, height, width)
        #
        #   Kspace mask is of size (batch_dim, n_channels, height, width )
        #

        mse = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

        kspace_true = self.image_to_kspace(image=y_true)
        kspace_pred = self.image_to_kspace(image=y_pred)

        # kspace_loss = tf.losses.mean_squared_error()

        kspace_loss = tf.abs(tf.subtract(kspace_true, kspace_pred))
        kspace_loss = tf.multiply(kspace_loss, kspace_mask)
        kspace_loss = tf.reduce_mean(tf.square(kspace_loss)) / tf.reduce_mean(
            tf.square(tf.abs(kspace_true))
        )

        # # (Batch, channels, height, widhth)
        # y_true_t = tf.transpose(y_true,perm=[0,2,3,1])
        # y_pred_t = tf.transpose(y_pred,perm=[0,2,3,1])
        # # NOW IT IS SIZE BATCH, HEIGHT, WIDTH, NUM_CHANNELS

        # Range of 0.5-0.001

        # ssim = tf.reduce_mean(tf.image.ssim_multiscale(
        #     img1=y_true,
        #     img2=y_pred,
        #     max_val=1
        #     ))

        # RANGE OF 0.4-0.008

        # total_variation = tf.reduce_mean(
        #     tf.image.total_variation(
        #         images=y_pred)
        #     )
        # 4k max

        # dy_true,dx_true = tf.image.image_gradients(image=y_true)
        # dy_pred,dx_pred = tf.image.image_gradients(image=y_pred)

        # #####
        # #####   X AND Y DERIVATIVES
        # #####

        # ## Make conv kernels

        # dx_filter_np = np.array([
        #     [-1,-2,-1],
        #     [0,0,0],
        #     [1,2,1]])

        dy_filter_np = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        # dx_filter = tf.constant(dx_filter_np,dtype=self.dtype,shape=(3,3,1,1))
        dy_filter = tf.constant(dy_filter_np, dtype=self.dtype, shape=(3, 3, 1, 1))

        # dx_filter_np = np.array([
        #     [-1],
        #     [0],
        #     [1]])

        # dy_filter_np = np.array([
        #     [-1,0,1],
        #     ])

        # dx_filter = tf.constant(dx_filter_np,dtype=self.dtype,shape=(3,1,1,1))
        # dy_filter = tf.constant(dy_filter_np,dtype=self.dtype,shape=(1,3,1,1))

        # # Apply convs to compute derivatives

        # dx_true = tf.nn.conv2d(
        #     input=y_true,
        #     filter=dx_filter,
        #     strides=(1,1,1,1),
        #     padding="VALID",
        #     data_format='NCHW')

        dy_true = tf.nn.conv2d(
            input=y_true,
            filter=dy_filter,
            strides=(1, 1, 1, 1),
            padding="VALID",
            data_format="NCHW",
        )

        # dx_pred = tf.nn.conv2d(
        #     input=y_pred,
        #     filter=dx_filter,
        #     strides=(1,1,1,1),
        #     padding="VALID",
        #     data_format='NCHW')

        dy_pred = tf.nn.conv2d(
            input=y_pred,
            filter=dy_filter,
            strides=(1, 1, 1, 1),
            padding="VALID",
            data_format="NCHW",
        )

        # # Compute derivatvies
        mse_dy = tf.losses.mean_squared_error(labels=dy_true, predictions=dy_pred)
        # # Range of 0.5-0.001

        # mse_dx = tf.losses.mean_squared_error(
        #     labels=dx_true,
        #     predictions=dx_pred)

        ## LAPLACIAN LOSS

        # MSE DY + DX is 0.0069

        # Central difference operator:
        # [-1 0 1]
        # OR SOBEL
        # [ -1 -2 -1
        # 0 0 0
        # 1 2 1]

        # laplacian_filter_np = np.array([
        #     [-1,-1,-1],
        #     [-1,8,-1],
        #     [-1,-1,-1]])

        # laplacian_filter_np = np.array([
        #     [0,-1,0],
        #     [-1,4,-1],
        #     [0,-1,0]])

        # # # NOTE: Conv2d takes input of (filter_height,filter_width,in_channels,out_channels)

        # lap_filt = tf.constant(laplacian_filter_np,dtype=self.dtype,shape=(3,3,1,1))

        # lap_true = tf.nn.conv2d(
        #     input=y_true,
        #     filter=lap_filt,
        #     strides=(1,1,1,1),
        #     padding="VALID",
        #     data_format='NCHW')

        # lap_pred = tf.nn.conv2d(
        #     input=y_pred,
        #     filter=lap_filt,
        #     strides=(1,1,1,1),
        #     padding="VALID",
        #     data_format='NCHW')

        # mse_lap = tf.losses.mean_squared_error(
        #     labels=lap_true,
        #     predictions=lap_pred)
        # Max of around 0.5

        # psnr = tf.reduce_mean(tf.image.psnr(
        #     a=y_true,
        #     b=y_pred,
        #     max_val=1))
        # Max of 12

        loss = mse + kspace_loss + mse_dy
        return loss


def main():
    """
        Tests the CNN.

    """
    parser = argparse.ArgumentParser(description='Please specify if you would like to use the center 2D slice or whole 3D volume for each scan')
    parser.add_argument('--2d', dest='run_2d', action='store_true')
    parser.add_argument('--3d', dest='run_2d', action='store_false')
    parser.set_defaults(run_2d=True)

    args = parser.parse_args()

    run_2d = args.run_2d

    '''
    if os.path.isdir("E:\\ENM_Project\\SPGR_AF2_242_242_1500_um\\"):
        study_dir = "E:\\ENM_Project\\SPGR_AF2_242_242_1500_um\\"
        project_folder = "E:/ENM_Project/"
    elif os.path.isdir("/run/media/bellowz"):
        study_dir = "/run/media/bellowz/S*/ENM_Project/SPGR_AF2_242_242_1500_um/"
        project_folder = "/run/media/bellowz/Seagate Backup Plus Drive/ENM_Project/"
    '''
    # output_folder = "b{}_e{}_se_{}_vs_{}".format(str(batch_size),str(epochs),
    #                                    str(steps_per_epoch),str(validation_steps))

    batch_size = 10
    acc_factor = 2
    max_epoch = 200
    polyfit = 4
    lr = 1e-4
    # name = 'test_kspace_loss'

    name = "b_{}_af_{}_e_{}_pf_{}_lr_{}".format(
        str(batch_size), str(acc_factor), str(max_epoch), str(polyfit), str(lr)
    )

    convnet = CNN(
        bool_2d=run_2d,
        #project_folder=project_folder,
        batch_size=batch_size,
        max_epoch=max_epoch,
        model_name=name,
        learn_rate=lr,
        acceleration_factor=acc_factor,
        polyfit_order=polyfit,
    )

    convnet.train()


if __name__ == "__main__":
    main()
