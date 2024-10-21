import cv2
import numpy as np
import dlib
import math

# ----------------------------------------------------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------------------------------------------------
predictor = dlib.shape_predictor("predictor/shape_predictor_68_face_landmarks.dat")     # Landmarks detector
detectFace = dlib.get_frontal_face_detector()                                           # Face detector object
LANDMARKS_N = 68        # Number of landmarks
CENTER_INDEX = 2        # Index of the center landmark
BLUR_MASK_SIZE = 5      # Gaussian kernel size
BLUR_SIGMA_X = 0        # Gaussian kernel standard deviation in X direction
THRESHOLD_VALUE = 60    # Threshold value for binarization
SPATIAL_MASK_SIZE = 3   # Spatial kernel size
BLACK_VALUE = 0
WHITE_VALUE = 255


# ----------------------------------------------------------------------------------------------------------------------
# Public functions
# ----------------------------------------------------------------------------------------------------------------------
def detect_faces(image):
    """
    Function that takes an image as input, and returns a list of faces, where the first one is the nearest to the cam

    :param image: input image
    :return: a list of faces, where the first one is the nearest to the cam
    """
    faces = detectFace(__get_gray_image(image))  # Detect faces in the image
    ret = np.array(faces)
    if len(ret) > 1:  # If there are more than one face
        for i in np.arange(1, len(ret)):  # Loop over the faces
            if __get_face_area(ret[i]) > __get_face_area(ret[0]):  # If the current face is bigger than the first one
                ret[i], ret[0] = ret[0], ret[i]  # Swap the faces
    return ret

def face_landmarks_detector(image, face):
    """
    Function that takes as input an image and a face, and returns a list of reference points of the face
    :param image: input image
    :param face: input face
    :return: a list of reference points of the face
    """
    reference_points = []
    landmarks = predictor(__get_gray_image(image), face)  # Landmarks predictor
    for i in range(0, LANDMARKS_N):
        point = (landmarks.part(i).x, landmarks.part(i).y)  # Get x,y coordinates of each landmark
        reference_points.append(point)
    return reference_points

def is_looking_at_cam(image, eye):
    """
    Function that determines if the eye is looking at the camera based on image and eye coordinates.
    
    :param image: Input image
    :param eye: Eye coordinates
    :return: True if the eye is looking at the camera, False otherwise
    """
    
    # Convert the image to grayscale
    image_grey = __get_gray_image(image)
    
    # Crop the eye from the image
    max_x = max(eye, key=lambda item: item[0])[0]
    min_x = min(eye, key=lambda item: item[0])[0]
    max_y = max(eye, key=lambda item: item[1])[1]
    min_y = min(eye, key=lambda item: item[1])[1]
    cropped_eye = image_grey[min_y:max_y, min_x:max_x]
    height, width = cropped_eye.shape

    # Apply spatial filtering to the eye
    if height > SPATIAL_MASK_SIZE * 2 and width > SPATIAL_MASK_SIZE * 2:
        margin = int((SPATIAL_MASK_SIZE - 1) / 2)
        cropped_eye = __averaging_filtering(cropped_eye, SPATIAL_MASK_SIZE)
        cropped_eye = cropped_eye[margin:(height - margin), margin:(width - margin)]
        height, width = cropped_eye.shape

    # Apply Gaussian blur to the eye
    if height > BLUR_MASK_SIZE and width > BLUR_MASK_SIZE:
        cropped_eye_blurred = cv2.GaussianBlur(cropped_eye, (BLUR_MASK_SIZE, BLUR_MASK_SIZE), BLUR_SIGMA_X)
        ret, threshold_eye = cv2.threshold(cropped_eye_blurred, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        ret, threshold_eye = cv2.threshold(cropped_eye, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply erosion and dilation to the eye for noise reduction
    p_size = int(height / 4) + 1
    threshold_eye_e = cv2.erode(threshold_eye, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p_size, p_size)))
    threshold_eye_d = cv2.dilate(threshold_eye_e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p_size, p_size)))

    # Find the largest component in the image
    threshold_eye = __largest_component_mask(threshold_eye_d)

    # Invert the binary image values
    threshold_eye = cv2.bitwise_not(threshold_eye)

    # Divide the eye into three parts
    div_part = int(width / 3)
    right_part = threshold_eye[0:height, 0:div_part]
    center_part = threshold_eye[0:height, div_part:div_part * 2]
    left_part = threshold_eye[0:height, div_part * 2:width]

    # Count black pixels in each part
    right_black_px = np.sum(right_part == 0)
    center_black_px = np.sum(center_part == 0)
    left_black_px = np.sum(left_part == 0)
    values = [right_black_px, left_black_px, center_black_px]

    # Check if the pupil of the eye is in the center of the image
    if values.index(max(values)) == CENTER_INDEX:
        for i in range(div_part):
            column = threshold_eye[0:height, div_part + i]
            if np.sum(column == WHITE_VALUE) < 2:
                return True
    return False


# ----------------------------------------------------------------------------------------------------------------------
# Private functions
# ----------------------------------------------------------------------------------------------------------------------
def __midpoint(cord1, cord2):
    """
    Function that takes two pairs of coordinates and returns the midpoint of the segment between them.
    
    :param cord1: first pair of coordinates
    :param cord2: second pair of coordinates
    :return: the midpoint of the segment between the two pairs of coordinates
    """
    x1, y1 = cord1
    x2, y2 = cord2
    return int((x1 + x2) / 2), int((y2 + y1) / 2)

def __euclidean_distance(cord1, cord2):
    """
    Function that takes two pairs of coordinates and returns the Euclidean distance between them.
    
    :param cord1: first pair of coordinates
    :param cord2: second pair of coordinates
    :return: the Euclidean distance between the two pairs of coordinates
    """
    x1, y1 = cord1
    x2, y2 = cord2
    return math.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))

def __get_gray_image(image):
    """
    Function that takes an image as input and returns its gray version.

    :param image: input image
    :return: gray version of the input image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def __get_face_area(face):
    """
    Function that takes a face as input and returns its area.

    :param face: input face
    :return: area of the face
    """
    return (face.right() - face.left()) * (face.bottom() - face.top())

def __averaging_filtering(image, mask_size):
    """
    Apply an averaging spatial filter to the input image.

    :param image: Binary input image.
    :param mask_size: Size of the mask. Must be an odd number.
    :return: Image after applying the averaging spatial filter.
    """
    return cv2.blur(image, (mask_size, mask_size))

def __largest_component_mask(image):
    """
    Function that finds the largest component in a binary image and returns the component as a mask.
    
    This function processes a binary image to identify and isolate its largest connected component. 
    It does so by finding contours in the image and selecting the one with the greatest area. 
    The largest contour is then drawn on a new image, effectively creating a mask of the largest component.

    :param image: input binary image
    :return: mask of the largest component in the image
    """
    max_area = [0, 0]  # (contour_index, area)

    # Find all external contours in the binary image
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    # Iterate through each contour to find the largest one
    for i, contour in enumerate(contours):
        contour_area = cv2.moments(contour)['m00']  # Calculate the area of the contour
        if contour_area > max_area[1]:  # Check if this contour is the largest found so far
            max_area[0] = i  # Store the index of the largest contour
            max_area[1] = contour_area  # Update the largest area

    # Create an empty image to draw the largest contour
    labeled_image = np.zeros(image.shape, dtype=np.uint8)
    
    # Draw the largest contour on the labeled image
    cv2.drawContours(labeled_image, contours, max_area[0], color=255, thickness=-1)

    return labeled_image  # Return the mask of the largest component
