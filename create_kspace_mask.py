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
#########       ADAPTED FROM SRIKANT IYER'S MATLAB CODE
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

def gen_pdf(img_size, poly_order=3, usf=0.5, dist_penalty=2, radius=0, total_iter=1e4):
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

        check_point = (min_offset + max_offset) / 2  # Reset point of interest to avg
        # Between the upper and lower bounds

        pdf = np.power((1 - r), poly_order).flatten() + check_point
        # Create a candidate pdf by taking the weighted distance matrix
        # and adding the checked value to it

        pdf[pdf > 1] = 1  # Truncate to 1
        pdf[indices_to_keep] = 1  # Ensure prescribed center of kspace is sampled
        pdf = pdf.reshape((img_height, img_width))

        num_sampled_pts = np.floor(pdf.sum())

        if num_sampled_pts == num_desired_pts:
            # print("PDF CONVERGED ON ITERATION " + str(iter_num) + "\n\n")
            break
        if num_sampled_pts > num_desired_pts:  # Infeasible
            max_offset = check_point
        if num_sampled_pts < num_desired_pts:  # Feasible but not optimal
            min_offset = check_point

        iter_num += 1

    return pdf, check_point


def gen_sampling_mask(pdf, max_iter=150, sample_tol=0.5):
    # pdf -> The simulated pdf from gen_pdf
    #       NOTE: THIS pdf is not normalized, i.e.
    #               integral_-inf^inf != 0
    #   max_iter -> simply the maxiumum iterations
    #
    #   sample_tol -> the acceptable deviation away from the
    #           desired number of sample points

    total_elements = pdf.shape[0] * pdf.shape[1]  # Height times width = total pixels

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
            candidate_mask = np.random.uniform(low=0.0, high=1.0, size=total_elements)
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

def view_mask_and_pdfs(the_pdf,the_mask,acc_factor,flip_flag=False):

    if the_pdf.shape[0]==1:
        the_pdf=the_pdf.T 
        the_mask = the_mask.T
        flip_flag=True

    if the_pdf.shape[1]==1:
        n=the_pdf.shape[0]

        tmp_pdf = the_pdf.flatten()
        tmp_mask = the_mask.flatten()

        the_pdf = np.zeros((n,n))
        the_mask = np.zeros((n,n))

        for counter in range(n):
            the_pdf[:,counter]=tmp_pdf
            the_mask[:,counter]=tmp_mask

    if flip_flag:
        the_pdf = the_pdf.T 
        the_mask = the_mask.T 


    fig, axes =plt.subplots(1,2)
    axes[0].imshow(the_pdf,cmap='jet')
    axes[1].imshow(the_mask,cmap='gray')
    axes[0].set_title('PDF')
    axes[1].set_title('MASK')
    fig.suptitle(str(acc_factor)+'X kspace Acceleration')
    # plt.savefig(save_dir+str(acc_factor)+'x.png')
    plt.show()
    return


if __name__ == "__main__":

    # NOTE: READOUT IS FREQUENCY ENCODING
    # WE HAVE NO PENALTY IN THE READOUT DIRECTION
    # PHASE ENCODING IS LEFT-RIGHT, 


    save_dir = 'D:\\Desktop\\Class\\a_Spring_2020\\ENM_531\\Project\\'

    cc = 128
    # size_of_image = np.array([cc, cc])
    size_of_image = np.array([1,512])


    pdf1 , offset_value1 = gen_pdf(img_size=size_of_image, poly_order=5, usf=0.5, dist_penalty=2, radius=0)
    (mask1, sf1)=gen_sampling_mask(pdf1,max_iter=150,sample_tol=0.5)


    view_mask_and_pdfs(the_pdf=pdf1,the_mask=mask1,acc_factor=2)


    pdf2 , offset_value2 = gen_pdf(img_size=size_of_image, poly_order=5, usf=0.25, dist_penalty=2, radius=0)
    (mask2, sf2)=gen_sampling_mask(pdf2,max_iter=150,sample_tol=0.5)


    view_mask_and_pdfs(the_pdf=pdf2,the_mask=mask2,acc_factor=4)


    pdf3 , offset_value3 = gen_pdf(img_size=size_of_image, poly_order=5, usf=0.125, dist_penalty=2, radius=0)
    (mask3, sf3)=gen_sampling_mask(pdf3,max_iter=150,sample_tol=0.5)


    view_mask_and_pdfs(the_pdf=pdf3,the_mask=mask3,acc_factor=8)


    # Recommend L1 over L2
    # 
    # With anisotropy, you have preference in terms of which direction to acquire
    #   one has higher SNR. 
    # kx (read) and ky is same, kz is not same
    # L1 or modify the distance function in L2 norm distance parameter







    ############
    ############
    ############ TEST CODE FOR READING DICOM FILES IN PYTHON BELOW::
    ############
    ############


    # scan_dir = (
    #     "E:\\DL_dicom_mda_data\\SPGR_AF2_242_242_1500_um\\2012_7_10_Moseman_3T\\data\\"
    # )
    # dcm_files = glob.glob(scan_dir + "*.dcm")
    # # print(dcm_files)
    # dcm_fil = dcm_files[0]
    # img_fil = pydicom.dcmread(dcm_fil)

    # print(img_fil)

    # img = img_fil.pixel_array

    # plt.imshow(img,cmap='gray')
    # plt.show()

    # studydate = img_fil.StudyDate
    # seriesdate = img_fil.SeriesDate
    # acquisitiondate = img_fil.AcquisitionDate

    # series_description = img_fil.SeriesDescription
    # patient_name = img_fil.PatientName

    # patient_id = img_fil.PatientID
    # patient_birth = img_fil.PatientBirthDate
    # patient_height_meters = img_fil.PatientSize
    # patient_weight = img_fil.PatientWeight
    # mr_acquisition_type = img_fil.MRAcquisitionType
    # sequence_name = img_fil.SequenceName
    # slice_thickness = img_fil.SliceThickness
    # repitition_time = img_fil.RepetitionTime
    # echo_time = img_fil.EchoTime
    # imaged_nucles = img_fil.ImagedNucleus
    # num_averages = img_fil.NumberOfAverages
    # echo_numbers = img_fil.EchoNumbers
    # num_phase_encoding_steps = img_fil.NumberOfPhaseEncodingSteps
    # echo_train_length = img_fil.EchoTrainLength
    # pct_sampling = img_fil.PercentSampling
    # pct_phase_fov = img_fil.PercentPhaseFieldOfView
    # pixel_bandwidth = img_fil.PixelBandwidth
    # software_version = img_fil.SoftwareVersions
    # protocol_name = img_fil.ProtocolName
    # transmit_coil_name = img_fil.TransmitCoilName

    # in_plane_phase_encoding_direction = img_fil.InPlanePhaseEncodingDirection 
    #     # THIS IS THE IMPORTANT ONE


    # flip_angle = img_fil.FlipAngle
    # variable_flip_angle_flag = img_fil.VariableFlipAngleFlag
    # sar = img_fil.SAR
    # patient_position = img_fil.PatientPosition  # Ex. 'ffs' for feet first supine
    # rows = img_fil.Rows
    # columns = img_fil.Columns
    # pixel_spacing = img_fil.PixelSpacing

    # print('\n\n\n')
    # print(img_fil.dir("hase"))

    # print(in_plane_phase_encoding_direction)