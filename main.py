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
import sys
import glob
from tensorflow.keras.layers import Lambda
import argparse
import logging
from datetime import datetime, timezone

#############
#############
#############       CUSTOM DEPENDENCIES!!!!!
#############
#############

import model_architectures

# from model_architectures import unet_7_layers, simple_cnn
from data_loader_class import data_generator
from show_3d_images import show_3d_images

def get_datetime():
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S")

def get_logger(name):
    filename = 'dev_' + get_datetime() + '.log'
    log_format = "%(asctime)s %(name)s %(levelname)5s %(message)s"
    logging.basicConfig(level=logging.DEBUG,format=log_format,
                        filename=filename,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(name).addHandler(console)
    logging.basicConfig(filename='LOGFILE.log',filemode='w')

    return logging.getLogger(name)

class CNN:
    # class CNN(tf.Module):
    def __init__(
        self,
        save_dir,
        #project_folder,
        batch_size,
        max_epoch,
        model_name,
        logger,
        bool_2d=True, 
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
        if not logger:
            self.logger = get_logger('cnn')
            self.logger.info('CNN initialization')
        else:
            self.logger = logger

        tf.logging.set_verbosity(tf.logging.ERROR)
        tf.set_random_seed(seed=1)

        #self.project_folder = project_folder
        self.save_dir = save_dir

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
            bool_2d=self.bool_2d,
        )

        if self.bool_2d:
            (
                self.num_train,
                self.num_valid,
                self.num_channels,
                self.img_height,
                self.img_width,
                _,
                _,
            ) = self.my_gen.get_info()
        else:
            (
                self.num_train,
                self.num_valid,
                self.num_channels,
                self.img_height,
                self.img_width,
                self.num_slices,
                _,
                _
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

        if self.bool_2d:
            self.input_matrix_shape = (
                None,
                self.num_channels,
                self.img_height,
                self.img_width,
            )
        else:
            self.input_matrix_shape = (
                None,
                self.num_channels,
                self.img_height,
                self.img_width,
                self.num_slices
          )

        self.logger.info("Input Matrix Shape: " + str(self.input_matrix_shape))

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

        self.logger.info('Tensor input has shape: ' + str(tensor_input.shape))

        if self.bool_2d:
            tensor_output = model_architectures.unet_9_layers(tensor_input)
        else:
            tensor_output = model_architectures.unet_7_layers_3D(tensor_input)
        self.logger.info("Shape of tensor_output: " + str(tensor_output.shape))
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
        save_str = ''
        if self.save_dir != '':
            save_str = self.save_dir + self.model_name

        steps_per_epoch_train = int(np.ceil(self.num_train / self.batch_size))
        steps_per_epoch_valid = int(np.ceil(self.num_valid / self.batch_size))

        train_epoch_loss = []
        valid_epoch_loss = []

        start_time = timeit.default_timer()

        self.logger.info("NOW BEGIN TRAINING OF {} EPOCHS".format(self.max_epoch))

        for epoch_num in range(self.max_epoch):

            self.logger.info("\n\n EPOCH NUMBER " + str(epoch_num + 1))

            train_batch_loss = []
            valid_batch_loss = []


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

            self.logger.info(
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

            self.logger.info(
                "VALID ==> Epoch [%d/%d], Loss: %.12f, Time: %2fs"
                % (epoch_num + 1, self.max_epoch, valid_batch_loss, elapsed)
            )
            start_time = timeit.default_timer()

            valid_epoch_loss.append(valid_batch_loss)

            if save_str != '':
                if (epoch_num + 1) % 100 == 0:
                    self.logger.info("SAVING MODEL . . . ")
                    self.saver.save(self.sess, save_str, global_step=epoch_num + 1)
        if save_str != '':
            self.saver.save(self.sess, save_str, global_step=epoch_num + 1)

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
        
        '''DEBUG: Removing kspace mse for now
        kspace_true = self.image_to_kspace(image=y_true)
        kspace_pred = self.image_to_kspace(image=y_pred)

        # kspace_loss = tf.losses.mean_squared_error()

        kspace_loss = tf.abs(tf.subtract(kspace_true, kspace_pred))
        kspace_loss = tf.multiply(kspace_loss, kspace_mask)
        kspace_loss = tf.reduce_mean(tf.square(kspace_loss)) / tf.reduce_mean(
            tf.square(tf.abs(kspace_true))
        )
        '''

        if self.bool_2d:
            # ## Make conv kernels

            dy_filter_np = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

            dy_filter = tf.constant(dy_filter_np, dtype=self.dtype, shape=(3, 3, 1, 1))

            dy_true = tf.nn.conv2d(
                input=y_true,
                filter=dy_filter,
                strides=(1, 1, 1, 1),
                padding="VALID",
                data_format="NCHW",
            )

            dy_pred = tf.nn.conv2d(
                input=y_pred,
                filter=dy_filter,
                strides=(1, 1, 1, 1),
                padding="VALID",
                data_format="NCHW",
            )

            # # Compute derivatvies
            mse_dy = tf.losses.mean_squared_error(labels=dy_true, predictions=dy_pred)

            loss = mse + kspace_loss + mse_dy
        else:
            '''Fine tuning regularizers is last step
            gauss_filt_np = np.zeros((3,3,3))
            gauss_filt_np[:,:,0] = [[6, 10, 6],
                                  [10, 17, 10],
                                  [6, 10,6]]
            gauss_filt_np[:,:,1] = [[10, 17, 10],
                                  [17, 28, 17],
                                  [10, 17, 10]]
            gauss_filt_np[:,:,2] = gauss_filt_np[:,:,0]

            gauss_filt = tf.constant(gauss_filt_np, dtype=self.dtype, shape=(3,3,3,1,1))

            # orignal order = (batch, channels, height, width, depth)
            y_true_t = tf.transpose(y_true, perm=[0,1,4,2,3])
            y_pred_t = tf.transpose(y_pred, perm=[0,1,4,2,3])
            #  [batch, in_channels, in_depth, in_height, in_width]

            dy_dz_true = tf.nn.conv3d(
                input=y_true_t,
                filter=gauss_filt,
                strides=(1, 1, 1, 1, 1),
                padding="VALID",
                data_format="NCDHW",
            )

            dy_dz_pred = tf.nn.conv3d(
                input=y_pred_t,
                filter=gauss_filt,
                strides=(1, 1, 1, 1, 1),
                padding="VALID",
                data_format="NCDHW",
            )
            
            # # Compute derivatvies
            mse_dy_dz = tf.losses.mean_squared_error(labels=dy_dz_true, predictions=dy_dz_pred)
            '''
            loss = mse # + kspace_loss -- DEBUG: will add back if working

        return loss


def main():
    """
        Tests the CNN.

    """
    logger = get_logger('cnn')
    logger.info('Running cnn')
 
    parser = argparse.ArgumentParser(description='Please specify if you would like to use the center 2D slice or whole 3D volume for each scan')
    parser.add_argument('--2d', dest='run_2d', action='store_true')
    parser.add_argument('--3d', dest='run_2d', action='store_false')
    parser.set_defaults(run_2d=True)
    parser.add_argument('-d', '--save_dir', help='Relative directory for model directory')

    args = parser.parse_args()

    save_dir = ''
    if not args.save_dir:
        logger.warning('No save directory listed for models. Will not save models')
    else: 
        save_dir = os.path.join(os.getcwd(), args.save_dir)

        if not os.path.exists(save_dir):
            logger.warning('Save directory does not exist, attempt to create')
            try:
                os.makedirs(save_dir)
            except OSError as err:
                logger.error('Cannot create save directory, exiting')
                sys.exit(1)

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

    batch_size = 1
    acc_factor = 2
    max_epoch = 200
    polyfit = 4
    lr = 1e-4
    # name = 'test_kspace_loss'

    name = "b_{}_af_{}_e_{}_pf_{}_lr_{}".format(
        str(batch_size), str(acc_factor), str(max_epoch), str(polyfit), str(lr)
    )

    logger.info('Building cnn')

    convnet = CNN(
        logger=logger,
        bool_2d=run_2d,
        batch_size=batch_size,
        max_epoch=max_epoch,
        save_dir=save_dir,
        model_name=name,
        learn_rate=lr,
        acceleration_factor=acc_factor,
        polyfit_order=polyfit,
    )

    logger.info('CNN model built. Training network')

    convnet.train()


if __name__ == "__main__":
    main()
