import cv2
import numpy as np
import matplotlib.pyplot as plt
import logic
import ImageRecognizer as imgRec
import time

# ----------------------------------------------------------------------------------------------------------------------
# Colors
# ----------------------------------------------------------------------------------------------------------------------
COLOR_GREEN = (0, 255, 0)   
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 247, 255)
COLOR_BLACK = (0, 0, 0)

# ----------------------------------------------------------------------------------------------------------------------
# Texts & text information
# ----------------------------------------------------------------------------------------------------------------------
FACE_DETECTION_MULTIPLE = "Number of faces:"
FACE_DETECTION_ERROR = "No face detected"
INTERFACE_TITLE = "Camera Capture"
FONT = cv2.FONT_HERSHEY_COMPLEX     # Font used for the text
FONT_SCALE = 1                      # Font scale
FONT_SCALE_LOW = 0.5                # Font scale for lower text
PADDING_NAME = 35                   # Padding between the name and the rectangle
TEXT_TOP_PADDING = 50               # Padding between the top of the window and the text 
TEXT_THICKNESS = 1                  # Thickness of the text
TEXT_THICKNESS_BOLD = 2             # Thickness of the text bold
MARGIN_TEXT_NAME = 6                # Margin between the name and the rectangle

# ----------------------------------------------------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------------------------------------------------
FRAME_RATE = 30                     # Frame rate
RECT_THICKNESS = 2                  # Thickness of the rectangle
CHECK_TIME = 2                      # Check time

# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
def __get_pixel_text_size(text):
    """
    Function that takes a text as input, and returns its length in pixels.
    
    This function is used to calculate the size of the text in pixels, 
    in order to have it centered on the screen.
    
    Parameters:
    text (str): input text
    
    Returns:
    tuple: A tuple containing the width and height of the text in pixels.
    """
    return cv2.getTextSize(text, FONT, FONT_SCALE, TEXT_THICKNESS)[0]

def __get_start_point_centered_text(text, width):
    """
    Function used to have the text centered according to the width of the window.
    
    This function takes a text and the width of the window as input, and returns the start point of the text
    in order to have it centered on the window.
    
    Parameters:
    text (str): input text
    width (int): window width
    
    Returns:
    int: The start point of the text
    """
    return int(width / 2 - __get_pixel_text_size(text)[0] / 2)

def __draw_rectangle_face(image, faces, name=""):
    """
    Function that takes an image as input, and returns the image with the faces identified.
    
    The nearest is inside a green rectangle, and the others, if there are, they are inside red rectangles
    :param image: input image
    :param faces: faces list
    :param name: username
    """
    # Draw a green rectangle around the nearest face
    cv2.rectangle(image, (faces[0].left(), faces[0].top()),
                  (faces[0].right(), faces[0].bottom()), COLOR_GREEN, RECT_THICKNESS)
    
    # Draw a label with a name below the face
    cv2.rectangle(image, (faces[0].left(), faces[0].bottom() - PADDING_NAME), (faces[0].right(), faces[0].bottom()),
                  COLOR_GREEN, cv2.FILLED)
    cv2.putText(image, name if name != "" else "?",
                (faces[0].left() + MARGIN_TEXT_NAME, faces[0].bottom() - MARGIN_TEXT_NAME), FONT,
                FONT_SCALE_LOW, COLOR_BLACK, TEXT_THICKNESS)
    
    # If there are more than one face, draw a red rectangle around them
    if len(faces) > 1:
        for i in np.arange(1, len(faces)):
            cv2.rectangle(image, (faces[i].left(), faces[i].top()),
                          (faces[i].right(), faces[i].bottom()), COLOR_RED, RECT_THICKNESS)
    
    # If the name is empty, display a message to register the name in the command line
    if name == "":
        cv2.putText(image, f'Register your name in the command line',
                    (__get_start_point_centered_text(FACE_DETECTION_ERROR, image.shape[1]), TEXT_TOP_PADDING * 7),
                    FONT, FONT_SCALE_LOW, COLOR_RED, TEXT_THICKNESS_BOLD)

def __handle_close(event, cap):
    """
    Handles the close event of the Matplotlib window by closing the camera capture.
    
    This function is called when the Matplotlib window is closed, and it is used to
    release the camera capture object.
    
    Parameters:
    event (matplotlib.backend_tkagg.KeyEvent): the close event
    cap (cv2.VideoCapture): the VideoCapture object to be closed
    """
    # Release the camera capture object
    cap.release()

def main():
    """
    This function initializes the face recognizer, camera, and starts an infinite loop to detect faces and show the results.
    """
    # Initializing the face recognizer.
    recognizer = imgRec.ImageRecognizer()

    # Initializing the camera.
    cap = cv2.VideoCapture(0)

    # Enabling the Matplotlib interactive mode.
    plt.ion()

    # Creating a figure to be updated.
    fig = plt.figure(INTERFACE_TITLE)

    # Intercepting the window's close event to call the __handle_close() function.
    fig.canvas.mpl_connect("close_event", lambda event: __handle_close(event, cap))

    # Preparing a variable for the first run.
    img = None

    # Initializing variable for last counted detected faces.
    last_time = 0
    last_n_detected_faces = 0

    # Initializing username variable.
    name = ""

    # Starting the infinite loop, as long as the camera remains open.
    while cap.isOpened():
        ret, frame = cap.read()
        if img is None:  # First run.
            # Showing the frame.
            img = plt.imshow(frame)
            plt.axis("off")
            plt.get_current_fig_manager().window.state('zoomed')  # Setting the window to full screen.
            plt.show()
        else:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = frame_gray.shape

            # Detecting if there are faces in the frame and eventually highlight with a colored square.
            faces = logic.detect_faces(frame)

            if len(faces) == 0:  # No detected faces case.
                cv2.putText(frame, f'No faces detected',
                            (__get_start_point_centered_text(FACE_DETECTION_ERROR, width), TEXT_TOP_PADDING),
                            FONT, FONT_SCALE, COLOR_RED, TEXT_THICKNESS_BOLD)

            if len(faces) > 0:  # Detected faces case.
                if (last_n_detected_faces != len(faces)) or (len(faces) > 1 and (time.time() - last_time) > CHECK_TIME):
                    last_time = time.time()
                    bounding_box = (faces[0].top(), faces[0].right(), faces[0].bottom(), faces[0].left())
                    name = recognizer.recognize_face(frame, bounding_box)
                
                reference_points = logic.face_landmarks_detector(frame, faces[0])  # Getting the reference

                # Getting the reference points of the right and left eye.
                right_eye = reference_points[36:42]
                left_eye = reference_points[42:48]

                # Verifying if the right eye and left eye are looking at the cam.
                is_looking_re = logic.is_looking_at_cam(frame, right_eye)
                is_looking_le = logic.is_looking_at_cam(frame, left_eye)
                
                # Drawing UI
                cv2.putText(frame, f'Faces detected: {len(faces)}',
                            (__get_start_point_centered_text(FACE_DETECTION_MULTIPLE, width), TEXT_TOP_PADDING),
                            FONT, FONT_SCALE, COLOR_YELLOW, TEXT_THICKNESS_BOLD)

                __draw_rectangle_face(frame, faces, name=name)
                cv2.putText(frame, f'Is looking?: ',
                            (__get_start_point_centered_text(FACE_DETECTION_ERROR, width), TEXT_TOP_PADDING * 2),
                            FONT, FONT_SCALE, COLOR_GREEN, TEXT_THICKNESS_BOLD)
                cv2.putText(frame, f'             {is_looking_le and is_looking_re}',
                            (__get_start_point_centered_text(FACE_DETECTION_ERROR, width), TEXT_TOP_PADDING * 2),
                            FONT, FONT_SCALE,
                            COLOR_GREEN if is_looking_le and is_looking_re else COLOR_RED, TEXT_THICKNESS_BOLD)

                # Setting the current image as the data to show
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img.set_data(image)
            else:
                # Setting the current frame as the data to show
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img.set_data(frame)
            last_n_detected_faces = len(faces)
            # Updating the figure associated to the shown plot
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(1 / FRAME_RATE)  # pause: 30 frames per second

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
