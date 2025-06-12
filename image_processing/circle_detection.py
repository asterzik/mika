import cv2
import numpy as np
from scipy import stats
import os
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QRect
from statsmodels.robust.scale import huber
import cProfile
import pstats

# from image_processing.hyspectral_images import denoise_hsi_images
import re
import time

import math


# TODO make that changeable in the GUI
trim_percentage = 0.05


def calculate_mean_intensity(
    image, mask, denoising_method, lower_threshold, upper_threshold
):
    values = image[mask > 0]
    if lower_threshold is not None and upper_threshold is not None:
        values = values[(values > lower_threshold) & (values < upper_threshold)]
    elif lower_threshold is not None:
        values = values[values > lower_threshold]
    elif upper_threshold is not None:
        values = values[values < upper_threshold]
    # If both are None, do not filter
    if len(values) == 0:
        return np.nan  # or 0, or handle as needed

    if denoising_method == "Mean":
        average = np.mean(values)
    elif denoising_method == "Median":
        average = np.median(values)
    elif denoising_method == "Mode":
        average = stats.mode(values)[0]
    elif denoising_method == "Trimmed Mean":
        n_trim = int(len(values) * trim_percentage)
        sorted_vals = np.sort(values)
        trimmed = sorted_vals[n_trim:-n_trim] if n_trim > 0 else sorted_vals
        average = np.mean(trimmed)
    elif denoising_method == "Huber Mean":
        average = huber(values)[0]
    else:
        average = np.mean(values)  # fallback

    return average


def compute_fore_back_ground_per_image(
    image,
    selected_circles,
    denoising_method,
    b_radius_inner,
    b_radius_outer,
    f_radius,
    shift,
    lower_threshold,
    upper_threshold,
):
    average_foreground = []
    average_background = []

    for pt in selected_circles:
        a, b, r = int(pt[0]), int(pt[1]), int(pt[2])
        a += int(round(shift[0]))
        b += int(round(shift[1]))

        # Calculate mean background intensity
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, (a, b), int(b_radius_outer), 255, -1)
        cv2.circle(mask, (a, b), int(b_radius_inner), 0, -1)
        background = calculate_mean_intensity(
            image, mask, denoising_method, lower_threshold, upper_threshold
        )
        average_background.append(background)
        # Calculate mean circle intensity
        circle_mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(circle_mask, (a, b), int(f_radius), 255, -1)
        foreground = calculate_mean_intensity(
            image, circle_mask, denoising_method, lower_threshold, upper_threshold
        )
        average_foreground.append(foreground)
    return np.array(average_foreground), np.array(average_background)


def compute_fore_back_ground_pixels(
    image,
    selected_circles,
    b_radius_inner,
    b_radius_outer,
    f_radius,
    shift,
    lower_threshold,
    upper_threshold,
):
    foreground_pixels = []
    background_pixels = []

    for pt in selected_circles:
        a, b, r = int(pt[0]), int(pt[1]), int(pt[2])
        a += int(round(shift[0]))
        b += int(round(shift[1]))

        # Background mask (ring)
        mask_bg = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask_bg, (a, b), int(b_radius_outer), 255, -1)
        cv2.circle(mask_bg, (a, b), int(b_radius_inner), 0, -1)
        background_pixels.append(image[mask_bg == 255])

        # Foreground mask (inner circle)
        mask_fg = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask_fg, (a, b), int(f_radius), 255, -1)
        foreground_pixels.append(image[mask_fg == 255])

    # Convert lists to numpy arrays
    foreground_pixels = np.concatenate(foreground_pixels)
    background_pixels = np.concatenate(background_pixels)
    if lower_threshold is not None and upper_threshold is not None:
        foreground_pixels = foreground_pixels[
            (foreground_pixels > lower_threshold)
            & (foreground_pixels < upper_threshold)
        ]
        background_pixels = background_pixels[
            (background_pixels > lower_threshold)
            & (background_pixels < upper_threshold)
        ]
    elif lower_threshold is not None:
        foreground_pixels = foreground_pixels[foreground_pixels > lower_threshold]
        background_pixels = background_pixels[background_pixels > lower_threshold]
    elif upper_threshold is not None:
        foreground_pixels = foreground_pixels[foreground_pixels < upper_threshold]
        background_pixels = background_pixels[background_pixels < upper_threshold]
    # If both are None, do not filter

    return foreground_pixels, background_pixels


class Circles:
    def __init__(
        self, image_path, dp, min_dist, param1, param2, min_radius, max_radius, parent
    ):
        """
        Initialize the Circles class with parameters and load images from the provided folder ("image_path").
        The average image of all images in the folder will be computed and stored in self.input_img.

        Parameters:
        - image_path (str): Path to the folder containing image files.
        - dp (float): Inverse ratio of the accumulator resolution to the image resolution.
        - min_dist (float): Minimum distance between the centers of detected circles.
        - param1 (float): Higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
        - param2 (float): Accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected.
        - min_radius (int): Minimum radius of the circles to detect.
        - max_radius (int): Maximum radius of the circles to detect.
        """
        # Initialize parameters
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.image_path = image_path
        self.detected = False
        self.detected_circles = None
        self.extinction_bool = False
        self.parent = parent
        self.selected_spot = None
        self.selected_spots = []
        self.selected_spot_labels = []
        self.foreground = None
        self.background = None

        self.load_images(image_path)

    def compute_circles(self):
        """
        Compute circles using the Hough Transform with the method 'HOUGH_GRADIENT' from cv2 and the instances parameters.

        Returns:
        - None
        """
        # gray = cv2.cvtColor(self.input_img, cv2.COLOR_BGR2GRAY)
        image = self.input_img.copy()
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert to 8-bit (normalize if needed)
        if image.dtype != np.uint8:
            # Normalize to 0–255 and convert to uint8
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = image.astype(np.uint8)

        # # Blur the image
        # gray_blurred = cv2.blur(gray, (3, 3))

        # Apply Hough transform to detect circles
        detected_circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        # Store detected circles
        if detected_circles is not None:
            self.detected_circles = np.uint16(np.around(detected_circles))
            return True
        else:
            return False

    def update_parameters(self, dp, min_dist, param1, param2, min_radius, max_radius):
        """
        Update circle detection parameters.

        Parameters:
        - dp (float): Inverse ratio of the accumulator resolution to the image resolution.
        - min_dist (float): Minimum distance between the centers of detected circles.
        - param1 (float): First method-specific parameter.
        - param2 (float): Second method-specific parameter.
        - min_radius (int): Minimum radius of the circles to detect.
        - max_radius (int): Maximum radius of the circles to detect.

        Returns:
        - None
        """
        # Update parameters
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.detected = False

    def update_min_dist(self, min_dist):
        """
        Update spot detection min_dist parameter.

        Parameters:
        - min_dist (float): Minimum distance between the centers of detected circles.

        Returns:
        - None
        """
        # Update parameters
        self.min_dist = min_dist
        self.detected = False

    def update_min_radius(self, min_r):
        self.min_radius = min_r
        self.detected = False

    def update_max_radius(self, max_r):
        self.max_radius = max_r
        self.detected = False

    def load_images(self, image_path, denoising_method="None"):
        """
        Load images and group them by frame, then sort them by wavelength.
        Compute average image
        update self.image_names, self.wavelengths, self.frames, self.images, self.input_img

        Parameters:
        - image_path (str): Path to the folder containing image files.

        Returns:
        - None
        """

        # Load image names from the directory
        self.image_names = [
            f
            for f in os.listdir(image_path)
            if f.endswith((".png", ".jpg", ".jpeg", ".tiff"))
        ]

        # Initialize lists for wavelengths, frames, and a dictionary for grouped images
        self.wavelengths = []
        self.frames = []
        grouped_images = {}

        # TODO Get the pattern extraction right, when they decided on a format
        def extract_wl_frame(filename):
            filename, _ = filename.split(".")  # Remove file extension
            patterns = [
                re.compile(r"(\d+)_(\d+)"),  # Format 1
                re.compile(r"imageAtLED(\d+)Frame(\d+)"),  # Format 2
                re.compile(r"imLCTFatWL(\d+)Frame(\d+)"),  # Format 3
            ]

            for i, pattern in enumerate(patterns):
                match = pattern.search(filename)
                if match:
                    wl, frame = match.groups()
                    # Check if the files were generated with a LED device and convert LED indices to wavelengths
                    if i == 0:
                        mapping = [470, 500, 530, 590, 615, 660]
                        wl = mapping[int(wl)]
                    if i == 1:
                        mapping = [470, 500, 530, 560, 590, 615, 660]
                        wl = mapping[int(wl)]

                    return int(wl), int(frame)

            raise ValueError(f"Pattern not implemented yet for filename: {filename}")

        # Group images by frame
        for image_name in self.image_names:
            w, f = extract_wl_frame(
                image_name
            )  # Extract wavelength and frame using regex
            wavelength = int(w)
            frame = int(f)

            # Append wavelength and frame to respective lists
            self.wavelengths.append(wavelength)
            self.frames.append(frame)

            # Initialize a new list for the frame if it's not already in grouped_images
            if frame not in grouped_images:
                grouped_images[frame] = []

            # Load the grayscale image
            # image = cv2.imread(
            #     os.path.join(image_path, image_name), cv2.IMREAD_GRAYSCALE
            # )
            # image = cv2.imread(
            #     os.path.join(image_path, image_name), cv2.IMREAD_UNCHANGED
            # )
            # if image is None:
            #     raise ValueError(f"Failed to load image: {image_name}")
            # if image.ndim == 3:
            #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image_raw = cv2.imread(
                os.path.join(image_path, image_name), cv2.IMREAD_UNCHANGED
            )

            if image_raw is None:
                raise ValueError(
                    f"Failed to load image: {image_name}. Check file path and integrity."
                )

            # Convert the image to float32 immediately to preserve precision.
            # This handles both 8-bit and higher bit-depth images (e.g., 16-bit TIFFs)
            # and ensures subsequent operations are done with higher precision.
            image_float = image_raw.astype(np.float32)

            # If the image was loaded as a color image (e.g., BGR, BGRA), convert it to grayscale.
            # This conversion will now operate on float32 data, resulting in a float32 grayscale image.
            if image_float.ndim == 3:
                # Check for alpha channel (4 channels)
                if image_float.shape[2] == 4:
                    image = cv2.cvtColor(image_float, cv2.COLOR_BGRA2GRAY)
                else:  # Assume BGR (3 channels)
                    image = cv2.cvtColor(image_float, cv2.COLOR_BGR2GRAY)
            else:
                # Image was already grayscale (2D array)
                image = image_float

            # Append the (wavelength, image, image_name) tuple to the list for the frame
            grouped_images[frame].append((wavelength, image, image_name))

        # Show a warning if no image files were found
        if not self.image_names:
            QMessageBox.warning(
                self, "Warning", "No image files found in the selected folder."
            )
            return

        # Prepare to store the final images and their names in the desired order
        self.images = np.empty(
            (len(np.unique(self.wavelengths)), len(np.unique(self.frames))),
            dtype=object,
        )
        self.image_names = np.empty_like(self.images)

        # Iterate through grouped images, process them, and append to self.images and self.image_names
        for frame, images_with_wavelengths in grouped_images.items():
            # Sort images by wavelength within each frame
            images_with_wavelengths.sort(
                key=lambda x: x[0]
            )  # Sort by the wavelength (first element of the tuple)

            # Extract the sorted grayscale images and their names
            sorted_images = [img for _, img, _ in images_with_wavelengths]

            smoothed_images = sorted_images

            for i, (wavelength, image, image_name) in enumerate(
                images_with_wavelengths
            ):
                self.images[i, frame - 1] = smoothed_images[i]
                self.image_names[i, frame - 1] = image_name

        # Update self.image_names to reflect the new order
        self.image_names = self.image_names.flatten().tolist()
        self.images = self.images.flatten().tolist()
        self.wavelengths.sort()

        # Iterate over all images for the first frame
        unique_wl = np.unique(self.wavelengths)
        unique_frames = np.unique(self.frames)
        highest_contrast_image = None
        highest_contrast_wl = None
        highest_contrast = 0
        # Only check for the first frame, the other frames are similar
        index_frame = unique_frames.tolist().index(0)
        self.first_frame_images = []
        for wl in unique_wl:
            index_wavelength = unique_wl.tolist().index(wl)
            image = self.images[index_wavelength * len(unique_frames) + index_frame]
            self.first_frame_images.append(image)

            std = np.std(image)
            if std > highest_contrast:
                highest_contrast = std
                highest_contrast_wl = wl
                highest_contrast_image = image

        self.input_img = highest_contrast_image

        # Compute pixels shifts relative to the representative image
        self.shifts = np.zeros((len(unique_wl), 2))

        for wl in unique_wl:
            if wl == highest_contrast_wl:
                continue
            index_wavelength = unique_wl.tolist().index(wl)
            image = self.images[index_wavelength * len(unique_frames) + index_frame]

            shift, _ = cv2.phaseCorrelate(highest_contrast_image, image)
            self.shifts[index_wavelength] = shift

        num_repeats = int(len(self.frames) / len(unique_frames))
        self.frames = np.tile(unique_frames, num_repeats).tolist()

    def get_shift(self, wl):
        unique_wl = np.unique(self.wavelengths)
        index_wavelength = unique_wl.tolist().index(wl)
        return self.shifts[index_wavelength]

    def get_frames(self):
        return np.unique(self.frames)

    def get_wavelengths(self):
        return np.unique(self.wavelengths)

    def get_image(self, wavelength, frame):

        index_wavelength = np.unique(self.wavelengths).tolist().index(wavelength)
        unique_frames = np.unique(self.frames).tolist()
        index_frame = unique_frames.index(frame)
        image = self.images[index_wavelength * len(unique_frames) + index_frame]

        image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Convert the image to QImage
        height, width = image_uint8.shape
        bytes_per_line = width
        q_image = QImage(
            image_uint8.copy().data,
            width,
            height,
            bytes_per_line,
            QImage.Format_Grayscale8,
        ).rgbSwapped()

        pixmap = QPixmap.fromImage(q_image)
        return pixmap

    def get_representative_image(self):
        """
        Get the input image as a QPixmap object.

        Returns:
        - QPixmap: QPixmap object containing the input image.
        """
        display_image_uint8 = cv2.normalize(
            self.input_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )

        height, width = display_image_uint8.shape
        bytes_per_line = width

        # Use .copy().data to ensure QImage takes ownership of the data buffer,
        # preventing potential issues if the original numpy array's memory is released.
        qimg = QImage(
            display_image_uint8.copy().data,
            width,
            height,
            bytes_per_line,
            QImage.Format_Grayscale8,
        )

        qpixmap = QPixmap.fromImage(qimg)
        return qpixmap

    def fore_background_first_time_step(self, b_radius_inner, b_radius_outer, f_radius):
        lower_threshold = None
        if self.parent.lower_threshold_enabled.isChecked():
            lower_threshold = self.parent.lower_threshold_param.value()
        upper_threshold = None
        if self.parent.upper_threshold_enabled.isChecked():
            upper_threshold = self.parent.upper_threshold_param.value()

        foreground_list = []
        backkground_list = []

        pool = mp.Pool(mp.cpu_count())

        mp_inputs = [
            (
                image,
                self.detected_circles[0, self.selected_spots],
                b_radius_inner,
                b_radius_outer,
                f_radius,
                self.shifts[np.unique(self.wavelengths).tolist().index(wl)],
                lower_threshold,
                upper_threshold,
            )
            for image, wl in zip(
                self.first_frame_images, np.unique(self.wavelengths), strict=True
            )
        ]
        with ThreadPool(processes=4) as pool:
            results = pool.starmap(compute_fore_back_ground_pixels, mp_inputs)
        pool.close()
        pool.join()
        foreground, background = zip(*results)
        foreground = np.concatenate(foreground)
        background = np.concatenate(background)
        return foreground, background

    def detected_circles(self):
        """
        Get the detected circles.

        Returns:
        - numpy.ndarray: Array containing detected circles of shape (1, num_circles, 3).
                         each circle is defined by its center and radius (x,y,r).
        """
        if not self.detected:
            self.detected = self.compute_circles()
        return self.detected_circles

    def draw_circles(self):
        """
        Draw circles on the input image around detected points.

        If circles are not already detected, it computes circles before drawing them.
        """
        # Compute circles if not already detected
        if not self.detected:
            self.detected = self.compute_circles()

        # Iterate over detected circles and draw them on the image
        self.parent.interactive_image.displayRepresentativeImage()
        if self.detected:
            for pt in self.detected_circles[0, :]:
                self.parent.interactive_image.highlight_circle_by_coordinates(
                    pt[0], pt[1], pt[2]
                )
            self.extinction_bool = False

        return self.parent.interactive_image.pixmap_item.pixmap()

    def draw_image(self, image):
        # Convert the numpy array image to a QPixmap object for display
        height, width, channels = image.shape
        bytes_per_line = channels * width
        qimg = QImage(
            image, width, height, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        qpixmap = QPixmap.fromImage(qimg)
        return qpixmap

    def add_or_remove_circle(self, x, y):
        """
        Add or remove a circle based on the provided coordinates (x, y).

        Parameters:
        - x (int): The x-coordinate of the mouse click.
        - y (int): The y-coordinate of the mouse click.

        Returns:
        - None

        Notes:
        - This function adds a new circle if there are no detected circles or if the
        mouse click is not close enough to any existing circles.
        - If there are detected circles and the mouse click is close enough to one
        of them, that circle will be removed.
        """
        if self.detected and self.detected_circles.shape[1] > 0:
            # Compute distances of (x, y) to the detected circles:
            distances = np.sqrt(
                np.sum(
                    (self.detected_circles[:, :, :2] - np.array([x, y])) ** 2, axis=2
                )
            )
            dist = np.min(distances)
            min_index = np.argmin(distances)
            # Delete closest circle if we clicked into (or close enough to) a circle
            if self.detected_circles[:, min_index, 2] * 1.3 >= dist:
                self.detected_circles = np.delete(
                    self.detected_circles, min_index, axis=1
                )
            else:
                # Create a new circle at the average radius of existing circles
                new_circle = np.array(
                    [
                        [
                            [
                                int(x),
                                int(y),
                                int(np.around(np.mean(self.detected_circles[:, :, 2]))),
                            ]
                        ]
                    ]
                )
                self.detected_circles = np.append(
                    self.detected_circles, new_circle, axis=1
                )
        else:
            # Create a new circle with a default radius
            self.detected_circles = np.array(
                [[[int(x), int(y), int((self.min_radius + self.max_radius) * 0.5)]]]
            )
            self.detected = True

        # Redraw the circles after adding or removing
        return self.draw_circles()

    def select_spot(self, x, y, group=0):
        if self.detected:
            # Compute distances of (x, y) to the detected circles:
            distances = np.sqrt(
                np.sum(
                    (self.detected_circles[:, :, :2] - np.array([x, y])) ** 2, axis=2
                )
            )
            dist = np.min(distances)
            min_index = np.argmin(distances)
            self.selected_index = min_index
            # Delete closest circle if we clicked into (or close enough to) a circle
            radius = self.detected_circles[:, min_index, 2]
            if radius * 1.3 >= dist:
                self.selected_spot = self.detected_circles[0, min_index, :]
                self.selected_spots.append(min_index)
                self.selected_spot_labels.append(group)
        self.parent.parent.spot_selection_changed = True

    def select_spots(self, rect, group=0):
        circles = self.detected_circles[0]
        if self.detected:
            # Find closest point in the rectangle to each circle:
            closest_x = np.clip(
                circles[:, 0], rect.topLeft().x(), rect.bottomRight().x()
            )
            closest_y = np.clip(
                circles[:, 1], rect.topLeft().y(), rect.bottomRight().y()
            )

            # Compute distances to the closest point in the rectangle
            distances = np.sqrt(
                (closest_x - circles[:, 0]) ** 2 + (closest_y - circles[:, 1]) ** 2
            )

            # # Check which circles overlap with the rectangle
            # overlap_indices = np.where(distances <= circles[:, 2])[0]
            # Check where the center of the circlesw overlaps with the rectangle

            overlap_indices = np.where(distances <= 0)[0]

            preselected = np.where(np.isin(self.selected_spots, overlap_indices))[0]

            # Filter out preselected indices from overlap_indices if their group is similar to the current group
            new_overlap_indices = [
                idx for idx in overlap_indices if idx not in self.selected_spots
            ]

            spots_to_add = []
            labels_to_add = []
            for index in reversed(preselected):
                spot = self.selected_spots.pop(index)
                labels = self.selected_spot_labels.pop(index)
                if group in labels:
                    if len(labels) == 1:
                        continue
                    else:
                        labels = [l for l in labels if l != group]
                else:
                    labels.append(group)
                labels_to_add.append(labels)
                spots_to_add.append(spot)

            self.selected_spots.extend(spots_to_add)
            self.selected_spots.extend(new_overlap_indices)

            self.selected_spot_labels.extend(labels_to_add)
            group_labels = np.ones_like(new_overlap_indices) * group
            self.selected_spot_labels.extend(
                [[group] for group in group_labels.tolist()]
            )
        self.parent.parent.spot_selection_changed = True

    def deselect_spots(self):
        self.selected_spots = []
        self.selected_spot_labels = []
        self.draw_circles()
        self.parent.parent.spot_selection_changed = True

    def compute_extinction(self):

        # Compute circles if not already detected
        if not self.detected:
            self.detected = self.compute_circles()

        n_images = len(self.images)
        n_circles = len(self.selected_spots)
        spots = self.detected_circles[0, self.selected_spots]

        self.foreground = np.zeros((n_images, n_circles))
        self.background = np.zeros((n_images, n_circles))

        pool = mp.Pool(mp.cpu_count())

        lower_threshold = None
        if self.parent.lower_threshold_enabled.isChecked():
            lower_threshold = self.parent.lower_threshold_param.value()
        upper_threshold = None
        if self.parent.upper_threshold_enabled.isChecked():
            upper_threshold = self.parent.upper_threshold_param.value()

        mp_inputs = [
            (
                image,
                spots,
                self.parent.parent.mean_method,
                self.parent.background_inner_radius_param.value(),
                self.parent.background_outer_radius_param.value(),
                self.parent.inner_radius_param.value(),
                self.shifts[np.unique(self.wavelengths).tolist().index(wl)],
                lower_threshold,
                upper_threshold,
            )
            for image, wl in zip(self.images, self.wavelengths, strict=True)
        ]
        t0 = time.time()
        with ThreadPool(processes=4) as pool:
            results = pool.starmap(compute_fore_back_ground_per_image, mp_inputs)
        print(f"Multiprocessing (starmap) took {time.time() - t0:.2f} seconds")
        pool.close()
        pool.join()
        self.foreground, self.background = zip(*results)
        self.foreground = np.array(self.foreground)
        self.background = np.array(self.background)
        self.parent.draw_histogram()

        self.extinction = -1 * np.log10(self.foreground / self.background)
        self.extinction_bool = True

    def get_extinction(self):
        if len(self.selected_spots) == 0:
            number_spots = np.shape(self.detected_circles)[1]
            self.selected_spots = list(range(number_spots))
            self.selected_spot_labels = [[0] for _ in range(number_spots)]
            for i in range(number_spots):
                self.selected_spot = self.detected_circles[0, i]
                self.parent.interactive_image.highlight_circle(self.selected_spot)

        return (
            np.array(self.frames),
            np.array(self.wavelengths),
            self.extinction,
        )

    def select_all_spots(self, image, group=0):
        number_spots = np.shape(self.detected_circles)[1]
        self.selected_spots = list(range(number_spots))
        self.selected_spot_labels = [[group] for _ in range(number_spots)]
        for i in range(number_spots):
            self.selected_spot = self.detected_circles[0, i]
            image.highlight_circle(self.selected_spot, [group])
        self.parent.parent.spot_selection_changed = True

    def highlight_selected(self, image, shift=None):
        # self.parent.interactive_image.displayAverageImage()
        for i, j in enumerate(self.selected_spots):
            self.selected_spot = self.detected_circles[0, j]
            group = self.selected_spot_labels[i]
            image.highlight_circle(self.selected_spot, group, shift)

    def getSelectedSpotIndices(self):
        return np.array(self.selected_spots)

    def getSelectedSpotLabels(self):
        return self.selected_spot_labels

    def cleanup(self):
        """Clean up resources held by this object."""

        # Clean up detected circles
        if self.detected_circles is not None:
            del self.detected_circles
            self.detected_circles = None

        # Clean up images and input_img
        if self.images is not None:
            del self.images
            self.images = None

        if self.input_img is not None:
            del self.input_img
            self.input_img = None

        # Clean up foreground and background extinction data
        if self.foreground is not None:
            del self.foreground
            self.foreground = None

        if self.background is not None:
            del self.background
            self.background = None

        # Clean up selected spots and labels
        self.selected_spots.clear()
        self.selected_spot_labels.clear()

        # Optional: clear parent reference (depending on use case)
        self.parent = None
