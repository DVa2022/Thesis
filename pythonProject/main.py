import os
from os import listdir
from os.path import isfile, join
import imageio.v2
import pydicom as di
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage
import scipy
import findpeaks
import scipy.signal
from scipy import ndimage
import skimage.feature
import math
import pandas as pd
from PIL import Image

# [8:393, 147:774]
# f = open("C:/Users/David/PycharmProjects/thesisData/labels.json")
# labels = json.load(f)
# for annotation in labels["annotations"]:  # composed of xmin, ymin, width, height
#     annotation["bbox"][0] = annotation["bbox"][0] - 147  # xmin of the bounding box
#     annotation["bbox"][1] = annotation["bbox"][1] - 8  # ymin of the bounding box
# with open("C:/Users/David/PycharmProjects/thesisData/labels2.json", "w") as outfile:
#     json.dump(labels, outfile)

# import pylbfgs

# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def plots_of_statistics():
    # The function creates variable of statistics of the data and plots them

    names_of_suitable_videos, _, _, _, _, _ = get_list_of_suitable_videos_in_drive()
    sexlist = []  # list of the sexes of all videos
    agelist = []  # list of the ages of all videos
    shapes_list = []  # list of the shapes of the numpy arrays of the videos
    cine_rate_list = []  # list of the number of frames per second
    effective_durations_list = []  # list of the  lengths in time of the videos
    datelist = []
    hourlist = []
    IDlist = []
    pixel_size_list = []
    counter = 0  # used to show the progress of the loop
    for fragments in names_of_suitable_videos:
        path = os.path.join('E:', 'All_DCMs', fragments[0], fragments[1], fragments[2])
        clip2 = di.dcmread(path)
        sexlist += [clip2.PatientSex]
        agelist += [clip2.PatientAge]
        shapes_list += [clip2.pixel_array.shape]
        cine_rate_list += [clip2.CineRate]
        effective_durations_list += [clip2.EffectiveDuration]
        datelist += [[clip2.AcquisitionDate[:4], clip2.AcquisitionDate[4:6], clip2.AcquisitionDate[6:]]]
        hourlist += [clip2.AcquisitionTime[:2]]
        IDlist += [clip2.PatientID]
        try:
            pixel_size_list += [clip2.SequenceOfUltrasoundRegions[0].PhysicalDeltaY]
        except:
            pixel_size_list += [-10]
        counter += 1
        if counter % 50 == 1:
            print(counter)

    age_count = []  # used to count the number of videos in each age
    for age in range(0, 120):
        age_count += [agelist.count(str(age))]
    # plt.plot(age_count)
    plt.bar(range(0, 120), age_count)
    plt.show()

    # some histograms:
    # plt.hist(np.array(effective_durations_list).astype(int))

    # pi charts:
    # labels, count = np.unique(np.array(shapes_list)[:, 0], return_counts=True)
    # fig, ax = plt.subplots()
    # ax.pie(count, labels=labels)
    # plt.show()

    # for pixel_size_list
    # pixel_size_array = np.array(pixel_size_list)
    # pixel_size_array[pixel_size_array == -10] = np.mean(pixel_size_array[pixel_size_array != -10])
    # hist = plt.hist(pixel_size_array * 10, 30)
    # plt.xlabel('Pixel size in mm')
    # plt.show()

    # for hourlist:
    # labels, count = np.unique(np.array(hourlist), return_counts=True)
    # count = np.insert(count, 1, np.zeros((7)))
    # plt.bar(range(0, 24), count)
    # plt.show()

    # for datelist
    # from collections import Counter
    # date_array = np.array(datelist)
    # date_array = date_array.astype(int)
    # date_array2 = date_array[:, :2]
    # date_list = [tuple(x) for x in date_array2]
    # date_counts = Counter(date_list)
    # sorted_dates = sorted(date_counts.keys())
    # x = ['{}-{}'.format(month, year) for year, month in sorted_dates]
    # y = [date_counts[date] for date in sorted_dates]
    # plt.bar(x, y)
    # plt.xticks(rotation=45)
    # plt.show()

    # getting the number of scans in each A### folder
    # alldirs = os.listdir('E:\All_DCMs')
    # scans_per_patient_list = []
    # for dir in alldirs:
    #     all_subdirs = os.listdir('E:/All_DCMs/' + dir)
    #     for subdir in all_subdirs:
    #         try:
    #             scans_per_patient_list += [len(os.listdir('E:/All_DCMs/' + dir + '/' + subdir))]
    #         except:
    #             continue
    # scans_per_patient_count = []
    # for num_of_scans in range(0, 84):
    #     scans_per_patient_count += [scans_per_patient_list.count(num_of_scans)]
    # # plt.plot(age_count)
    # plt.bar(range(0, 84), scans_per_patient_count)
    # plt.show()

    # # basically checking if there's one patientID field in the Dicom file for each real person:
    # sex_array = np.array(sexlist)
    # age_array = np.array(agelist)
    # ID_array = np.array(IDlist)
    # ID_list_unique_values = list(set(IDlist))  # a list all unique IDs found in the Dicom files. Each ID presumably corresponds to one person
    # multi_sex_count = 0  # the number of the patients who have both male and female written in their Dicom files. The number is 0
    # sex_and_no_sex_count = 0  # the number of the patients who have no sex in some of their Dicom files and do have a sex in other ones. There were 24 of these patients
    # male_count = 0  # the number of male patients. Around 109
    # female_count = 0  # the number of female patients. Around 75
    # no_sex_at_all_count = 0  # the number of patients with no written sex. Around 155
    # multiple_ages_count = 0  # the number of patients with more than one age written in their videos
    # num_of_scans_per_patient = []
    # for ID in ID_list_unique_values:
    #     sexes_of_patient = sex_array[np.where(ID_array == ID)]  # the sexes written in the Dicom files of a patient
    #     if 'M' in sexes_of_patient and 'F' in sexes_of_patient:
    #         multi_sex_count += 1
    #     if ('M' in sexes_of_patient and '' in sexes_of_patient) or ('F' in sexes_of_patient and '' in sexes_of_patient):
    #         sex_and_no_sex_count += 1
    #     if 'M' in sexes_of_patient:
    #         male_count += 1
    #     if 'F' in sexes_of_patient:
    #         female_count += 1
    #     if 'M' not in sexes_of_patient and 'F' not in sexes_of_patient:
    #         no_sex_at_all_count += 1
    #     ages_of_patient = age_array[np.where(ID_array == ID)]
    #     if np.all(ages_of_patient == ages_of_patient[0]):
    #         multiple_ages_count += 1
    #     num_of_scans_per_patient += [np.where(ID_array == ID)[0].size]
    # scans_per_patient_list = num_of_scans_per_patient
    # scans_per_patient_count = []
    # max_scans_per_patient = np.max(np.array(num_of_scans_per_patient))
    # for num_of_scans in range(0, max_scan_per_patienr):
    #     scans_per_patient_count += [scans_per_patient_list.count(num_of_scans)]
    # plt.bar(range(0, max_scans_per_patient), scans_per_patient_count)
    # plt.show()

    # some histograms:
    # all_xmins, all_ymins, all_widths, all_heights = [], [], [], []
    # for annotation in labels["annotations"]:  # composed of xmin, ymin, width, height
    #     xmin, ymin, width, height = annotation["bbox"]
    #     all_xmins += [xmin]
    #     all_ymins += [ymin]
    #     all_widths += [width]
    #     all_heights += [height]
    # all_xmins = np.array(all_xmins)
    # all_ymins = np.array(all_ymins)
    # all_widths = np.array(all_widths)
    # all_heights = np.array(all_heights)
    # hist = plt.hist(all_xmins, bins=30)
    # plt.xlabel("left corner position of bounding box in pixels")
    # plt.show()
    # hist = plt.hist(all_ymins, bins=30)
    # plt.xlabel("upper corner position of bounding box in pixels")
    # plt.show()
    # hist = plt.hist(all_widths, bins=30)
    # plt.xlabel("Width in pixels")
    # plt.show()
    # hist = plt.hist(all_heights, bins=30)
    # plt.xlabel("Height in pixels")
    # plt.show()

    # pixel_size_array[pixel_size_array == -10] = np.mean(pixel_size_array)  # there's one video where it didn't get the pixel size so the value of -10 was assigned
    # this code doesn't work because some videos have more than one bounding box
    # hist = plt.hist(all_xmins * pixel_size_array * 10, bins=30)
    # plt.xlabel("left corner position of bounding box in mm")
    # plt.show()
    # hist = plt.hist(all_ymins * pixel_size_array * 10, bins=30)
    # plt.xlabel("upper corner position of bounding box in mm")
    # plt.show()
    # hist = plt.hist(all_widths * pixel_size_array * 10, bins=30)
    # plt.xlabel("bounding box width in mm")
    # plt.show()
    # hist = plt.hist(all_heights * pixel_size_array * 10, bins=30)
    # plt.xlabel("bounding box height in mm")
    # plt.show()

    # I began writing this in order to fix the problem with the code above:
    # pixel_size_per_bb_list = []
    # for pixel_size in pixel_size_array:
    #

    np.save("age_count", np.array(age_count))
    np.save("agelist", np.array(agelist))
    np.save("sexlist", np.array(sexlist))
    np.save("shapes_list", np.array(shapes_list))
    np.save("cine_rate_list", np.array(cine_rate_list))
    np.save("effective_durations_list", np.array(effective_durations_list))
    np.save("datelist", np.array(datelist))
    np.save("hourlist", np.array(hourlist))
    np.save("IDlist", np.array(IDlist))
    np.save("pixel_size_list_with_one_negative_value", np.array(pixel_size_list))


def create_mask(image, threshold=0, phased_array=True, *args):
    """
    Creates a mask that keeps the biggest blob in the input image.
    :param image: a given image, needs to be grayscale, np.array.
    :param threshold: threshold for binary image, int.
    :param phased_array: whether the lung shape is a triangle= phased array transducer, type: bool.
    :param args: minimum width size (previously: minimum blob size), int.
    :return:
        the mask, np.array.
    """

    binary_image = np.zeros_like(image, dtype='uint8')
    binary_image[image > threshold] = 1
    # find all your connected components (white blobs in your image)
    n_components, output, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    mask = binary_image.copy()
    if args:
        min_w = args
        for i in range(1, n_components):  # first component is background
            # for every component in the image, you delete it if it's below min_size *width*
            if stats[i, 2] < min_w:
                mask[output == i] = 0
    else:
        comp_sizes = stats[1:, -1]  # without background component
        min_size = np.max(comp_sizes)
        for i in range(1, n_components):  # first component is background
            # for every component in the image, you keep it only if it's above min_size
            if stats[i, -1] < min_size:
                mask[output == i] = 0

    if np.sum(mask) >= image.size * 0.9:  # if the mask covers almost all of the image
        mask = create_mask(image, threshold=1, phased_array=phased_array, *args)
    # if phased_array:
    #     [y, x] = np.nonzero(mask)
    #     x_left_bottom, x_mid_top, x_right_bottom = np.min(x), x[0], np.max(x)
    #     y_top, y_bottom = y[0], np.max(y)
    #     pt1, pt2, pt3 = [x_left_bottom, y_bottom], [x_mid_top, y_top], [x_right_bottom, y_bottom]
    #     triangle_cnt = np.array([pt1, pt2, pt3])
    #     tri_mask = np.zeros_like(mask)
    #     cv2.drawContours(tri_mask, [triangle_cnt], 0, 1, -1)
    #     mask *= tri_mask

    # mask = morphology.remove_small_objects(mask, min_size=5)
    # contours = cv2.findContours(mask, method=0, mode=0)

    if threshold == 1:
        contours = skimage.measure.find_contours(mask, 0.8)
        if len(contours) != 2:
            print("here")
        assert len(contours) == 2, "The number of contours isn't 2."
        if contours[0].shape < contours[1].shape:
            contour = contours[0]
        else:
            contour = contours[1]

        # Create an empty image to store the masked array
        triangle_mask = np.zeros_like(mask, dtype='bool')
        # Create a contour image by using the contour coordinates rounded to their nearest integer value
        triangle_mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
        # Fill in the hole created by the contour boundary
        triangle_mask = ndimage.binary_fill_holes(triangle_mask)
        triangle_mask = (~triangle_mask).astype(int)
        mask = mask * triangle_mask
    return mask


def clip_preprocessing(clip, out_size=(256, 384), is_phased_array=True):  # , preprocess_resize=False
    """
    Preprocesses the LUS clip - cropping it to create the ROI
    :param clip: LUS clip, type: np.ndarray.
    :param out_size: The required output size, type: tuple.
    :param is_phased_array: checks if the transducer is phased array - lung shape is triangle, type: bool.
    # :param preprocess_resize: whether or not to resize the clip's frames to [512,512], type: bool.
    :return:
        ROI_Clips, type: np.ndarray.
    """

    if clip.ndim == 3:  # converting 1-ch clip to 3-ch
        clip = np.stack((clip, clip, clip), axis=clip.ndim)
    assert clip.ndim == 4
    [frames, rows, _, _] = clip.shape

    first_frame = clip[0, :, :, 0]  # first frame- red channel
    lung_mask = create_mask(first_frame, threshold=0, phased_array=is_phased_array)

    assert out_size[0] <= rows
    # [y, x] = np.nonzero(lung_mask)
    # x_min, x_max = np.min(x), np.max(x)  # the triangle's base
    # y_min, y_max = np.min(y), np.max(y)
    # pix_below_lung = rows - y_max
    # pix_above_lung = out_size[0] - (y_max - y_min) - pix_below_lung
    # y_min -= pix_above_lung
    # making sure the width of the image is 'out_size[1]' pixels:
    # if x_max - x_min > out_size[1]:
    #     x_min += (x_max - x_min - out_size[1]) // 2
    #     x_max = out_size[1] + x_min
    # elif x_max - x_min < out_size[1]:
    #     x_min -= (out_size[1] - x_max + x_min) // 2
    #     x_max = out_size[1] + x_min

    # rgb_clip = []
    # lung_mask_rgb = gray2rgb(lung_mask)
    #
    # for k in range(frames):
    #     # if preprocess_resize:
    #     #     frame = cv2.resize(clip[k, :, :, :], [512, 512], interpolation=cv2.INTER_CUBIC)
    #     #     rgb_clip.append((frame[:360, :, :] * lung_mask_rgb)[:, x_min:x_max, :])
    #     # else:
    #     # rgb_clip.append((clip[k, 30:30 + out_size[0], :, :] * lung_mask)[:, x_min:x_max, :])
    #     frame = clip[k, ...] * lung_mask_rgb
    #     rgb_clip.append(frame[y_min:y_min + out_size[0], x_min:x_max, :])

    # cropped_mask = lung_mask[y_min:y_min + out_size[0], x_min:x_max]  # single channel

    # return cropped_mask
    return lung_mask


def create_circular_mask(h, w, center=None, radius=None, return_dist=False):
    # creates a mask that is used to crop the clip so the clip doesn't have redundant pixels from deep in the lungs
    # the function was taken from here:
    # https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array

    # return_dist: A flag. If it's true, the function returns a matrix of the distances of all pixels from the cone top

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    if return_dist:  # usually it's False
        return dist_from_center
    else:
        mask = dist_from_center <= radius
        return mask


def crop_clip(clip):
    # gets a mask (2D array) or a clip (3D array where the first dimension is frame number)
    # and returns the coordinates of its vertices

    if clip.ndim == 3:  # if it's a clip and not a mask
        [y, x] = np.nonzero(clip[0])
    elif clip.ndim == 2:  # if it's a mask
        [y, x] = np.nonzero(clip)
    else:
        assert clip.ndim == 3 or clip.ndim == 3, "Can't crop the mask/clip. Wrong number of dimensions"
    x_left_bottom, x_mid_top, x_right_bottom = np.min(x), x[0], np.max(x)
    y_top, y_bottom = y[0], np.max(y)
    return y_top, y_bottom, x_left_bottom, x_right_bottom


def get_indices_on_arc(meanimg, filtered, cutting_factor):
    # meanimg: 2D mask of a scan. Has only 0s and 1s.
    # filtered: 'meanimg' after being spatially filtered.
    # cutting_factor: percentage of the original radius. It determines the radius of the new arc.
    # k: A numpy array with the shape of number of points on the arc X 2. Each row contains the xy coordinates of a
    #    point on the arc.

    mask_for_arc = create_circular_mask(filtered.shape[0], filtered.shape[1], center=(int(filtered.shape[1] / 2), 0),
                                        radius=cutting_factor * filtered.shape[0])
    # the last non-zero value in each column in the mask represents the y coordinate of the arc in for a given x value
    c = mask_for_arc.shape[0] - np.argmax(mask_for_arc[::-1, :], axis=0) - 1
    # 1D array of values representing the y coordinates of the arc points
    k = np.vstack((c, np.linspace(0, c.shape[0] - 1, c.shape[0]).astype('int'))).T  # coordinates of the arc points
    all_arc_intensities = (meanimg != 0)[k[:, 0], k[:, 1]]
    # first = np.argmax(all_arc_intensities)  # first occurence of non-zero value in meanimg that is also on the arc
    # last = k.shape[0] - np.argmax(
    #     all_arc_intensities[::-1]) - 1  # last occurence of non-zero value in meanimg that is also on the arc
    # k = k[first:last]  # coordinates of the arc points that are on the ultrasound scan cone

    return k


def get_arc_intensities(original, filtered, cutting_factor=0.5, returnCoordinate=False, relativePosition=0):
    # original: 2D mask of a scan. Has only 0s and 1s.
    # filtered: 'original' after being spatially filtered.
    # cutting_factor: percentage of the original radius. It determines the radius of the new arc.
    # returnCoordinate: a flag. If it's true the return value of the function is a coordinate along an arc.
    #                   The coordinate is used later to create a line from the top of the scan to the coordinate.
    # relativePosition: The relative position of the coordinate along the arc. It's between 0 and 1.

    # The function gets these variables and returns the intensity values of the scan along an arc with a certain
    # radius that is not bigger than the original scan radius. The arc resides only on the scan so on meaningless
    # zero values.
    # If returnCoordinate=True then the function returns the coordinate of a pixel on the arc.

    assert cutting_factor <= 1, "The cutting factor shouldn't be bigger than 1."

    k = get_indices_on_arc(original, filtered, cutting_factor)
    arc_intensities = filtered[k[:, 0], k[:, 1]]

    # plt.imshow(filtered, cmap='gray')
    # plt.scatter(k[:, 1], k[:, 0], color="red", s=4)
    # plt.show()

    # plt.plot(arc_intensities)
    # plt.show()

    if returnCoordinate:
        absolute_position = relativePosition * arc_intensities.size
        if relativePosition.size > 1:  # if there are two minima or more
            relative_position_on_last_arc = absolute_position.astype('uint16')
        else:  # if there's one minimum
            relative_position_on_last_arc = int(absolute_position)
        return k[relative_position_on_last_arc, :]

    return arc_intensities


def find_arcs_minima(original, filtered, beg=0.7, end=0.5, step=-0.02, return_result=False):
    # Calculates the locations of the minima of along an array which has a representative value of a few arcs in the
    # phased array scan.

    # beg: largest arc cutting factor. Which means it's the longest arc.
    # end: smallest arc cutting factor. Which means it's the shortest arc.
    # step: difference between the cutting factor value of each two consequent arc.
    # return_result: a flag used to change what's returned by the function.

    arc_intensities = get_arc_intensities(original, filtered, cutting_factor=beg)  # that's the longest arc

    size = arc_intensities.size
    stacked_arc_intensities = arc_intensities
    # creating many shorter arcs which are then resized and reshaped, so they're 1D:
    for cutting_factor in np.arange(beg + step, end, step):
        arc_intensities2 = get_arc_intensities(original, filtered, cutting_factor=cutting_factor)
        resized = cv2.resize(arc_intensities2, dsize=(1, size), interpolation=cv2.INTER_CUBIC)
        resized = np.reshape(resized, resized.size)
        stacked_arc_intensities = np.row_stack(
            (stacked_arc_intensities, resized))  # stacking them to use a median filter on them

    # show all interpolated/resized arcs together
    # plt.imshow(stacked_arc_intensities, cmap='gray')
    # plt.show()

    median_arc_intensities = np.median(stacked_arc_intensities, axis=0)  # creating a "representative" arc
    result = ndimage.median_filter(-1 * median_arc_intensities,
                                   size=15)  # filtering the arc so we won't get false local minima
    result = ndimage.gaussian_filter(result, sigma=17)
    if return_result:
        return result
    minima, _ = scipy.signal.find_peaks(result)

    # to show the same array many times, so it's easy to see it
    # rep = np.tile(median_arc_intensities, (17, 1))
    # plt.imshow(rep, cmap='gray')
    # plt.show()

    # result2 = cv2.equalizeHist((-1*result).astype('uint8'))

    # plt.plot(-1 * result)
    # plt.scatter(minima, -1 * result[minima], color="red")
    # thresh = np.percentile(-1 * result, 20)
    # plt.hlines(thresh, 0, size, color='red')
    # plt.show()

    return minima, minima / size


def showIm(image, pixel_size=None, title=None):
    # pixel_size: If it gets this parameter it means that the axes should be in centimeters.
    # title: This parameter is used only when pixel_size isn't None because if pixel_size is None the title is created
    #        outside this function.

    if pixel_size is None:
        plt.imshow(image, cmap='gray')
        plt.xlabel("Pixel number")
        plt.ylabel("Pixel number")
        plt.show()
    else:
        plt.clf()
        ysize = image.shape[0] * pixel_size
        xsize = image.shape[1] * pixel_size
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray', extent=[0, xsize, ysize, 0])
        plt.xlabel("cm")
        plt.ylabel("cm")
        plt.title(title)
        plt.show()

    return


def create_arc_mask(image, threshold=140, spacing=3, horizontal_pleura_flag=False):
    # horizontal_pleura_flag: a flag. When it's True, it means that the input is the reverse radon transformation and
    # the radii that define the arc mask output should be calculated differently

    if horizontal_pleura_flag:  # in this method (of this condition) the radii for mask_up and mask_down are found
        # according to the distances of the pixels of the iradon pleura (=image) from the top of the scan cone
        dist_from_center = create_circular_mask(image.shape[0], image.shape[1], center=(int(image.shape[1] / 2), 0),
                                                radius=image.shape[0], return_dist=True)
        dist_from_center = dist_from_center * (
                    image != 0)  # creating an image where it's 0 everywhere except in the pixels
        # of horizontal pleura, where the values are the distances from
        short_dist = math.floor(np.min(dist_from_center[dist_from_center > 0])) - spacing
        long_dist = math.ceil(np.max(dist_from_center)) + spacing
    else:  # and in this method (of this condition) the radii for mask_up and mask_down are found according to their
        # coordinates and not their
        n_components, output, stats, _ = cv2.connectedComponentsWithStats((image > threshold).astype('uint8'),
                                                                          connectivity=8)
        ind = np.argsort(stats[:, 4])[
            -2]  # getting the index of the component with the 2nd largest area. The largest is the background
        top_y = stats[ind, 1]  # row of the top pixel of the component
        down_y = top_y + stats[ind, 3] - 1  # row of the bottom pixel of the component
        left_x = stats[ind, 0]  # column of the leftmost pixel of the component
        right_x = stats[ind] - 1  # row of the rightmost pixel of the component
        top_x = int(np.mean(np.argwhere((output == ind)[top_y])))  # getting a column for the top pixel
        down_x = int(np.mean(np.argwhere((output == ind)[down_y])))  # getting a column for the bottom pixel
        point1 = np.array([top_y - spacing, top_x])  # taking 3 pixels above the upper pixel
        point2 = np.array([down_y + spacing, down_x])  # taking 3 pixels below the bottom pixel
        half_im_width = int(image.shape[1] / 2)
        point3 = np.array([0, half_im_width])  # coordinates of head of the scan cone
        short_dist = np.linalg.norm(
            point1 - point3)  # radius from the scan head to the uppermost pixel in the component
        long_dist = np.linalg.norm(
            point2 - point3)  # radius from the scan head to the bottommost pixel in the component

    mask_up = create_circular_mask(image.shape[0], image.shape[1], center=(int(image.shape[1] / 2), 0),
                                   radius=short_dist)
    mask_down = create_circular_mask(image.shape[0], image.shape[1], center=(int(image.shape[1] / 2), 0),
                                     radius=long_dist)
    arc_mask = np.logical_xor(mask_up, mask_down)
    # showIm(arc_mask)
    return arc_mask


def detect_ridges(gray, sigma=1.0):
    # This function was taken from here:
    # https://stackoverflow.com/questions/48727914/how-to-use-ridge-detection-filter-in-opencv
    # It applies a ridge detection algorithm that uses the eigenvalues/vectors of the Hessian of the image.
    # gray: The image.
    # sigma: "Standard deviation used for the Gaussian kernel, which is used as weighting function for the auto-correlation matrix".
    H_elems = skimage.feature.hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = skimage.feature.hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges


def stretch_histogram(img):
    # Gets an image and stretches its histogram between 0 and 255

    return (img - np.min(img)) / (np.max(img) - np.min(img)) * 255


def find_pleura(meanimg, filtered, anti_false_top_ridges_mask, Hessian_sigma=3.0, spacing=3):
    # meanimg: The image before filtering. It's used only to create a mask.
    # filtered: The image after filtering. It's used to find the pleura.
    # pleura: A binary mask of the pleural line

    pleura_too_big_flag = False

    maxima_ridges, minima_ridges = detect_ridges(filtered, sigma=Hessian_sigma)  # getting the ridges of the image
    filtered_only_scan_cone = filtered * (meanimg != 0)
    med = np.median(filtered_only_scan_cone[filtered_only_scan_cone > 0])
    arc_mask = create_arc_mask(filtered_only_scan_cone, threshold=med, spacing=4)
    minima_ridges[minima_ridges.shape[0] - 15:, :] = 0  # getting rid of some fake ridges in the middle of the last rows
    masked_ridges = -1 * arc_mask * minima_ridges * (meanimg != 0)  # getting rid of pixels that are far from the pleura
    masked_ridges = stretch_histogram(masked_ridges)
    masked_ridges = masked_ridges * arc_mask  # because of the histogram stretching there are some pixels that should be
    # set to zero by the mask.
    meanimg_mask = meanimg.copy()
    meanimg_mask[meanimg > 0] = 1
    masked_ridges = masked_ridges * meanimg_mask  # setting some more values to zero
    threshold = np.percentile(filtered[filtered > 0], 92)
    brightest_ridges = masked_ridges > threshold
    while np.sum(brightest_ridges) / np.size(brightest_ridges) > 0.055:
        threshold += 5
        brightest_ridges = masked_ridges > threshold

    for i in range(5):  # iteratively make the pleural line mask smaller, but only if it's too big in the 1st iteration
        if pleura_too_big_flag:
            threshold += 7

        brightest_ridges = masked_ridges > threshold
        brightest_ridges_copy = masked_ridges > threshold

        brightest_ridges = eliminate_ridges_above_shadows(brightest_ridges, meanimg, filtered)
        n_components, output, stats, _ = cv2.connectedComponentsWithStats(brightest_ridges.astype('uint8'),
                                                                          connectivity=8)

        horizontal_ridges = get_horizontal_lines(brightest_ridges, degree_margin=25)
        for obj_num in range(1, n_components):
            only_one_object = output == obj_num
            if np.sum(
                    only_one_object * horizontal_ridges) == 0:  # if none of the pixels of the object is in the inverse radon transform
                output[output == obj_num] = 0  # then nullify the object

        without_top_ridges = output * anti_false_top_ridges_mask  # getting rid of some fake ridges in the middle of the top rows
        output = without_top_ridges if np.sum(
            without_top_ridges) != 0 else output  # prevents the possibility of remaining with zero detected objects

        # ignoring the ridges near the edges because these are artifacts of my image processing algorithm
        output[:, :18] = 0
        output[:, -18:] = 0

        # the next step is necessary for cases when the 2nd largest object was deleted because it wasn't horizontal enough
        n_components, output, stats, _ = cv2.connectedComponentsWithStats(output.astype('uint8'),
                                                                          connectivity=8)
        ind = np.argsort(stats[:, 4])[
            -2]  # getting the index of the component with the 2nd largest area. The largest is the background

        ind = get_index_of_lowest_component_among_2_largest(n_components, stats)  # index of the lower component among
        # the two largest non-background components

        pleura = output == ind
        temp_result = horizontal_ridges * pleura  # mask of the "horizontal" iradon transform of the pleura
        if np.sum(temp_result) != 0:  # which is what almost always happens
            horizontal_pleura = temp_result
            arc_mask_for_found_pleura = create_arc_mask(horizontal_pleura, spacing=4, horizontal_pleura_flag=True)
            # showIm(arc_mask_for_found_pleura)
            pleura = pleura * arc_mask_for_found_pleura  # if the pleura and a B-line were merged, get rid of the vertical
            # component (= the B-line) of the merged object.

        if np.sum(pleura) / np.size(
                pleura) > 0.012:  # if the pleural line is unreasonably big (covers more than 1.2% of the frame)
            # TODO Find a smarter, more general, method to select a value for the above condition
            print(np.sum(pleura) / np.size(pleura))
            pleura_too_big_flag = True
            # plt.title('The pleura was too big so it was shrunk.')
        else:
            break

    removed_by_brightest_ridges_func = brightest_ridges_copy * 1 - brightest_ridges * 1
    result = 0.5 * pleura + 0.5 * brightest_ridges + 0.25 * removed_by_brightest_ridges_func
    # showIm(result)
    return pleura


def find_brightest_region(meanimg, filtered, percentile=98.2):
    spacing = 8
    bright_pixels_threshold = np.percentile(filtered[filtered > 0], percentile)
    bright_pixels_mask = filtered > bright_pixels_threshold
    n_components, output, stats, _ = cv2.connectedComponentsWithStats(bright_pixels_mask.astype('uint8'),
                                                                      connectivity=8)
    ind = np.argsort(stats[:, 4])[
        -2]  # getting the index of the component with the 2nd largest area. The largests is the background
    mask = output == ind
    # top_y = stats[ind, 1]  # row of the top pixel of the component
    # down_y = top_y + stats[ind, 3] - 1  # row of the bottom pixel of the component
    # left_x = stats[ind, 0] - spacing  # column of the leftmost pixel of the component
    # right_x = left_x + stats[ind, 2] - 1 + spacing  # row of the rightmost pixel of the component
    # boxed = np.zeros(meanimg.shape)
    # boxed[top_y:down_y, left_x:right_x] = filtered[top_y:down_y, left_x:right_x]
    # cropped_box = boxed[top_y:down_y, left_x:right_x]
    return mask


def prevent_side_ridges(meanimg, filtered):
    # meanimg: 2D numpy array which represents the time-filtered frame.
    # filtered: Spatially-filtered meanimg.

    # It's a preprocessing stage. It gets the scan frame, meanimg, and makes the dark regions have the same values as
    # the nearest non-black pixel. By doing that we won't get any ridges on the edges of the scan cone when a ridge
    # detection algorithm is used.

    # y coordinate of the left and right corners of the scan cone:
    y_coordinate_of_corners = np.nonzero(meanimg[:, 0])[0][0]
    # coordinates of the pixels of two lines - one from the top of the scan to the left and right corners respectively
    rr_left, cc_left = skimage.draw.line(int(filtered.shape[1] / 2), 0, 0, y_coordinate_of_corners)
    rr_right, cc_right = skimage.draw.line(int(filtered.shape[1] / 2), 0, meanimg.shape[1], y_coordinate_of_corners)
    meanimg_copy = meanimg.copy()

    k = get_indices_on_arc(meanimg, filtered, cutting_factor=1)  # indices on the last arc of the scan cone
    for j in range(y_coordinate_of_corners - 1):  # filling the black values above the cone with non-black values
        meanimg_copy[j, :rr_left[j]] = meanimg_copy[j, rr_left[j]]
        meanimg_copy[j, rr_right[j]:meanimg.shape[1]] = meanimg_copy[j, rr_right[j] - 1]

    for i in range(meanimg.shape[1] - 1):  # filling the black values below the cone with non-black values
        meanimg_copy[k[i, 0]:, i] = meanimg_copy[k[i, 0] - 1, i]

    return meanimg_copy


def eliminate_ridges_above_shadows(brightest_ridges, meanimg, filtered):
    # brightest_ridges:

    beg = 0.9
    result = -find_arcs_minima(meanimg, filtered, beg=beg, end=0.7, step=-0.02, return_result=True)
    thresh = np.percentile(result, 25)
    threshed_result = result < thresh

    # num_of_dark_regions = 0
    # for i in range(threshed_result.size - 1):
    #     if threshed_result[i] != threshed_result[i + 1]:
    #         num_of_dark_regions += 1

    # list1 = threshed_result.tolist()
    # count_dups = [sum(1 for _ in group) for _, group in groupby(list1)]
    # length = len(count_dups)
    # if threshed_result[0] == True:
    #     num_of_dark_regions = math.ceil(length/2)
    # else:
    #     num_of_dark_regions = math.floor(length / 2)
    indices_of_dark_regions = threshed_result.nonzero()
    indices_of_dark_regions = indices_of_dark_regions[0]  # get the array from the tuple
    indices_of_dark_regions = np.rint(indices_of_dark_regions / beg)  # the indices correspond to the indices on an arc
    # with the length of 0.9 times the largest arc so there should be an adjustment made (the division) and then the
    # indices are rounded to the nearest integer
    # coordinates = []
    # coordinates.append(indices_of_dark_regions[0])
    # for i in range(len(indices_of_dark_regions)-1):
    #     if indices_of_dark_regions[i+1] - indices_of_dark_regions[i] > 1:
    #         coordinates.append(indices_of_dark_regions[i])
    #         coordinates.append(indices_of_dark_regions[i+1])
    # coordinates.append(indices_of_dark_regions[len(indices_of_dark_regions)-1])
    # arr = np.array(coordinates) / threshed_result.size
    k = get_indices_on_arc(meanimg, filtered, 1)
    # arr = (arr * k.shape[0]).astype(int)
    # all_coordinates = k[arr]
    for i, val in enumerate(k):
        if i % 2 == 0 and i in indices_of_dark_regions:
            rr, cc = skimage.draw.line(0, int(filtered.shape[1] / 2), k[i, 0],
                                       k[i, 1])
            brightest_ridges[rr, cc] = 0

    # cone_mask = meanimg != 0
    # all_coordinates = np.reshape(all_coordinates, (int(all_coordinates.shape[0]/2), 2, 2))
    # for two_corners_coordinates in all_coordinates:
    #     rr_left, cc_left = skimage.draw.line(int(filtered.shape[1]/2), 0, two_corners_coordinates[0,1], two_corners_coordinates[0,0])
    #     rr_right, cc_right = skimage.draw.line(int(filtered.shape[1]/2), 0, two_corners_coordinates[1,1], two_corners_coordinates[1,0])
    #     for j in range(y_coordinate_of_corners - 1):  # filling the black values above the cone with non-black values
    #         meanimg_copy[j, :rr_left[j]] = meanimg_copy[j, rr_left[j]]
    #         meanimg_copy[j, rr_right[j]:meanimg.shape[1]] = meanimg_copy[j, rr_right[j] - 1]
    return brightest_ridges


def y_derivative(masked_ridges):
    sobel_Y = cv2.Sobel(masked_ridges, cv2.CV_64F, 0, 1)
    sobel_Y_abs = np.uint8(np.absolute(sobel_Y))
    threshold = np.percentile(sobel_Y_abs[sobel_Y_abs > 0], 94)
    return sobel_Y_abs > threshold


def get_horizontal_lines(brightest_ridges, degree_margin=20):
    # brightest_ridges: 2D numpy array representing a mask of the brightest ridges
    # degree_margin: parameter used to nullify values in the sinogram so after the inverse radon transform
    # there will be only horizontal lines and lines with a slope of degree_margin angles more ot less than
    # a horizontal line

    radon = skimage.transform.radon(brightest_ridges)  # radon transform
    radon[:, 0:90 - degree_margin] = 0  # nullifying angles of lines which are too "vertical"
    radon[:, 90 + degree_margin:] = 0  # nullifying angles of lines which are too "vertical"
    iradon = stretch_histogram(skimage.transform.iradon(radon))  # inverse radon transform
    iradon = iradon > 180  # 180 is a value that was chosen after trial and error

    # iradon and brightest_ridges have a different number of columns so the solution is to make an image with the shape
    # of brightest_ridges and with the pixels of iradon in its center (brightest_ridges has more columns).
    # The solution is given here:
    center_col_1 = int(brightest_ridges.shape[1] / 2)
    center_col_2 = int(iradon.shape[1] / 2)
    center_shift = center_col_1 - center_col_2
    full_sized_iradon = np.zeros(brightest_ridges.shape)
    try:
        full_sized_iradon[:, center_col_1 - center_col_2:center_col_1 + center_col_2] = iradon
    except:
        print('Had to change the size of the input to the variable "full_sized_iradon"')
        full_sized_iradon[:, center_col_1 - center_col_2:center_col_1 + center_col_2 + 1] = iradon

    return full_sized_iradon


def get_index_of_lowest_component_among_2_largest(n_components, stats):
    # n_components: Number of components in the mask of the ridges
    # stats: Statistics about the components
    # ind: Among the 2 largest non-background components by area, that's index of the lower component

    assert n_components >= 2, "Expected at least 2 components to be found but there's only one which is the background"
    indices_sorted_by_size = np.argsort(
        stats[:, 4])  # the indices of the detected components, sorted from the smallest to the largest, area-wise
    index_of_second_largest = indices_sorted_by_size[-2]
    if n_components >= 3:  # if there are at least 3 components, including the background
        index_of_third_largest = indices_sorted_by_size[-3]

        component_center_y_coordinate = stats[[index_of_second_largest, index_of_third_largest], 1] + 0.5 * stats[
            [index_of_second_largest,
             index_of_third_largest], 3]  # y coordinates of the centers of the second and third largest components
        height_index = np.argsort(component_center_y_coordinate)[
            -1]  # The index in component_center_y_coordinate of the lowest component
        if height_index == 0:  # if the largest non-background component is also the lower one, save it
            ind = index_of_second_largest
        else:  # i.e. height_index == 1
            if abs(component_center_y_coordinate[0] - component_center_y_coordinate[
                1]) > 37:  # if the two components are too far away
                # from each other, like in the case of a pleura and one of its a-lines, then ignore the a-line
                ind = index_of_second_largest
            elif abs(component_center_y_coordinate[0] - component_center_y_coordinate[1]) < 6:  # if the two components
                # are very close, it means that these are two parts of the same pleura so save the larger part
                ind = index_of_second_largest
            else:
                ind = index_of_third_largest
    else:  # if there's only 1 non-background component
        ind = index_of_second_largest

    return ind


def get_list_of_suitable_videos_in_drive():
    # The function goes over a table that contains the details about all the videos and returns the names of the
    # ones that are upper scan videos that were taken with a phased array probe + returns in the same variable the rest
    # of the path of the video.
    # Also returns:
    # tagged_frames_list: list of the number of the labelled frame of each suitable video
    # max_y: numpy array of the maximal y coordinate value of a pleural line label
    # max_x: numpy array of the maximal x coordinate value of a pleural line label
    # min_y: numpy array of the minimal y coordinate value of a pleural line label
    # min_x: numpy array of the minimal x coordinate value of a pleural line label

    system_name = 'linux'
    #if 'Linux' in os.getcwd():
    if 'linux' in system_name:
        all_videos_csv_path = '/workspace/all_data_270222_2.csv'
        #all_videos_csv_path = '/home/stu16/all_data_270222_2.csv'
    else:
        all_videos_csv_path = 'C:/Users/David/Desktop/titulo/research/fromTheDrive/All_Ichilov_Backups/all_data_270222_2.csv'
    all_videos = pd.read_csv(all_videos_csv_path)
    boolean_df_upper_scan = all_videos['Position'] == 'Upper_Scan'
    boolean_df_phased_array_probe = (all_videos['TransducerData'] == 'S5-1\\UNUSED\\UNUSED') | (
                all_videos['TransducerData'] == 'S4-2\\UNUSED\\UNUSED')
    boolean_df_suitable_videos = boolean_df_phased_array_probe & boolean_df_upper_scan  # a boolean dataframe
    boolean_df_videos_with_pleural_line_label = all_videos[
                                                    'Pleural_line_y_2'] != 0  # a boolean dataframe of the videos that have a pleural line labeling of at least two points
    boolean_df_suitable_videos = boolean_df_suitable_videos & boolean_df_videos_with_pleural_line_label

    names_of_suitable_videos = all_videos.iloc[:, 0:3][boolean_df_suitable_videos].to_numpy()

    # for j, file_name in enumerate(names_of_suitable_videos[:,2]):  # adding the names '.dcm'
    #     names_of_suitable_videos[j] = names_of_suitable_videos[j] + '.dcm'

    names_of_suitable_videos[:, 2] = names_of_suitable_videos[:, 2] + '.dcm'

    # Second part of this function begins here
    tagged_frames_list = all_videos['Tagged_Frame'][boolean_df_suitable_videos].tolist()

    y_coordinates_arr = all_videos.iloc[:, 9:19][boolean_df_suitable_videos].to_numpy()
    x_coordinates_arr = all_videos.iloc[:, 19:30][boolean_df_suitable_videos].to_numpy()
    # To get the indices of the videos where all y coordinates of the pleura are the same
    # vvvv = np.where((max_y - min_y) == 0)
    # suitable_videos_indices = all_videos.iloc[:, 9:19][boolean_df_suitable_videos].index
    # videos_with_consistent_y_value = suitable_videos_indices[vvvv]
    max_y = np.max(y_coordinates_arr, axis=1).astype(
        int)  # the maximal y coordinate of each pleural label. One label per video
    max_x = np.max(x_coordinates_arr, axis=1).astype(int)
    y_coordinates_arr[y_coordinates_arr == 0] = 10000  # getting rid of the zeros in these two arrays, so...
    x_coordinates_arr[x_coordinates_arr == 0] = 10000
    min_y = np.min(y_coordinates_arr, axis=1).astype(int)  # ... I can find in the following two arrays the minimal...
    min_x = np.min(x_coordinates_arr, axis=1).astype(int)  # ... non-zero value of each row

    return names_of_suitable_videos, tagged_frames_list, max_y, max_x, min_y, min_x


def get_all_patient_names(external_driver_flag):
    # Not used anymore!!!!

    # external_driver_flag: If true, get the videos from the driver of all videos. If false, get the videos from my PC

    if external_driver_flag:
        DCMs = os.listdir('D:/')
        if '$RECYCLE.BIN' in DCMs:  # removing a folder that doesn't contain videos
            DCMs.remove('$RECYCLE.BIN')
        if 'System Volume Information' in DCMs:  # removing a folder that doesn't contain videos
            DCMs.remove('System Volume Information')
        if 'First_Screen.mat' in DCMs:  # removing a folder that doesn't contain videos
            DCMs.remove('First_Screen.mat')
        if 'US_Main_19.exe' in DCMs:  # removing a folder that doesn't contain videos
            DCMs.remove('US_Main_19.exe')
        if 'saved_files_DCM_19' in DCMs:  # removing a folder that doesn't contain videos
            DCMs.remove('saved_files_DCM_19')
        all_patient_names = []
        for dcm in DCMs:
            As = os.listdir('D:/' + dcm)
            for i in range(len(As)):
                As[i] = dcm + '/' + As[i] + '/'
            all_patient_names += As
    else:
        all_patient_names = ('DCM_36/A801/', 'DCM_36/A802/')

    if 'DCM_21/desktop.ini/' in all_patient_names:
        all_patient_names.remove('DCM_21/desktop.ini/')

    return all_patient_names


def get_file_list_and_folder_path(what_patient, external_driver_flag, suitable_videos_list=None):
    # Not used anymore!!!

    # external_driver_flag: If true, get the videos from the driver of all videos. If false, get the videos from my PC
    # what_patient: Gets values only in the form of 'DCM_36/A801/'
    # suitable_videos_list: The values is not None only if external_driver_flag == True. It's a list of all phased array
    # ... upper scan files

    if external_driver_flag:
        folder_path = 'D:/' + what_patient
        file_list = os.listdir(folder_path)
        # finding the values in file_list that are also in suitable_videos_list.
        file_list = list(set(file_list) & set(suitable_videos_list))
    else:
        path_of_folder_of_tagged_videos = 'C:/Users/David/Desktop/titulo/research/fromTheDrive/All_Ichilov_Backups/saved_files_DCM_36/Results_' + what_patient
        file_list = os.listdir(path_of_folder_of_tagged_videos)  # list of videos with tags
        folder_path = 'C:/Users/David/Desktop/titulo/research/fromTheDrive/Group1/' + what_patient  # path of folder with DICOMs
        # only_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]  # list of DICOM file names
        for j, file_name in enumerate(file_list):  # adding the names '.dcm'
            file_list[j] = file_list[j] + '.dcm'

    return file_list, folder_path


def main_func():
    external_driver_flag = True  # used to determine if I'm working on the videos from my PC or from the external driver
    suitable_videos_list = get_list_of_suitable_videos_in_drive() if external_driver_flag else None
    if external_driver_flag:
        suitable_videos_list, tagged_frames_list, max_ys, max_xs, min_ys, min_xs = suitable_videos_list
    # all_patient_names = get_all_patient_names(external_driver_flag=external_driver_flag)
    # i = 0
    # aaa = np.stack((suitable_videos_list, tagged_frames_list, max_ys, max_xs, min_ys, min_xs), axis=1)
    images = []
    for i in range(len(suitable_videos_list)):
        # for what_patient in all_patient_names:  # what_patient value example: 'DCM_19/A401/'
        # file list value example: ['hashed_1198195877.dcm']
        # folder path value example: 'D:/DCM_19/A401/'
        # file_list, folder_path = get_file_list_and_folder_path(what_patient, external_driver_flag, suitable_videos_list[:, 2])

        # for vid_name in file_list:
        # with open(folder_path + vid_name, 'rb') as infile:

        # i = i + 1
        vid_name = suitable_videos_list[i, 2]
        if suitable_videos_list[i, 0] != 'DCM_37':
            continue
        path = 'E:/All_DCMs/' + suitable_videos_list[i, 0] + '/' + suitable_videos_list[i, 1] + '/' + vid_name
        print(vid_name, i, tagged_frames_list[i])
        try:
            clip = di.dcmread(path)
        except:
            continue

        frame = clip.pixel_array[tagged_frames_list[i] - 1]
        clip = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(clip, (min_xs[i], min_ys[i]), (max_xs[i], max_ys[i]), color=(0, 0, 255), thickness=2)
        special_cropping_flag = True
        if special_cropping_flag:
            clip = np.concatenate((cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)[57:-150], clip[100:int(len(frame) / 2)]))
        else:
            clip[int(len(frame) / 2):] = clip[:int(len(frame) / 2)]
            clip[:int(len(frame) / 2)] = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)[:int(len(frame) / 2)]
        title = suitable_videos_list[i, 0] + '_' + suitable_videos_list[i, 1] + '_' + vid_name[:-4] + '_'
        cv2.imwrite("not_labeled_no_BB_only_DCM_37/" + title + str(tagged_frames_list[i]) + ".jpg", clip)
        continue
        labeled_frame = clip
        pixel_size = clip.SequenceOfUltrasoundRegions[0].PhysicalDeltaY
        min_pleural_depth_in_cm = 1.001 * 1.5  # 10.01 mm, according to here: https://www.sciencedirect.com/science/article/pii/S2005290118300554
        min_pleural_depth_in_pixel = int(min_pleural_depth_in_cm / pixel_size)

        clip = clip.pixel_array
        mask = clip_preprocessing(clip, out_size=(clip.shape[1], clip.shape[2]), is_phased_array=True)
        clip = clip * mask
        y_top, y_bottom, x_left_bottom, x_right_bottom = crop_clip(mask)
        clip = clip[:, y_top:y_bottom + 1, x_left_bottom:x_right_bottom + 1]
        depth = round(clip.shape[1] * pixel_size)  # Total scan depth. Typically 10-19cm
        cutting_factor = 5.2 / depth  # equals to 0.4 when the depth is 13 cm. We wish to leave only the upper 5.2 cm after cropping
        circular_mask = create_circular_mask(clip.shape[1], clip.shape[2], center=(int(clip.shape[2] / 2), 0, 0),
                                             radius=cutting_factor * clip.shape[1])
        clip = clip * circular_mask
        y_top, y_bottom, x_left_bottom, x_right_bottom = crop_clip(clip)
        clip = clip[:, y_top:y_bottom + 1, x_left_bottom:x_right_bottom + 1]
        top_cropping_mask = create_circular_mask(clip.shape[1], clip.shape[2], center=(int(clip.shape[2] / 2), 0, 0),
                                                 radius=min_pleural_depth_in_pixel)
        top_cropping_mask = ~top_cropping_mask * 1  # used to get rid of some tissue that is for sure not the pleura

        window_size = 5
        sum_of_ridges = np.zeros((int(clip.shape[0] - 5), clip.shape[1], clip.shape[2]))
        sum_of_outputs = np.zeros((int(clip.shape[0] - 5), clip.shape[1], clip.shape[2]))
        kernel = np.ones((2, 2), np.uint8)

        for i in range(0, int(clip.shape[0] - 5)):
            meanimg = np.mean(clip[i:i + 3], axis=0).astype('uint8')
            # meanimg = np.mean(clip[:], axis=0).astype('uint8')
            # if external_driver_flag:
            #     title = folder_path + vid_name[:-4] + ' and frame: ' + str(i * 3)
            # else:
            #     title = folder_path[:15] + '.../' + vid_name[:-4] + ' and frame: ' + str(i * 3)
            plt.title(title)
            filtered = findpeaks.stats.denoise(meanimg, method='bilateral', window=window_size)
            # filtered = meanimg
            filtered = prevent_side_ridges(meanimg, filtered)
            # filtered = findpeaks.stats.denoise(meanimg_without_black_regions, method='bilateral', window=1)
            pleura_mask = find_pleura(meanimg, filtered, top_cropping_mask, Hessian_sigma=3.0)
            pleura_mask = pleura_mask.astype(int) * 255
            row_num = pleura_mask.shape[0]
            filtered[pleura_mask != 0] = 255
            # plt.title(f'The average of frames: {i*3}:{i*3+3}')
            final = filtered * (meanimg != 0)
            images.append(final)
            # showIm(final, pixel_size, title=title)
            # break

        imageio.v2.mimsave(vid_name[:-4] + '_segmentation_video.gif', images)  # (..., duration=0.5)
        images = []

    imageio.v2.mimsave('trial' + '.gif', images)  # (..., duration=0.5)

    print('g')
