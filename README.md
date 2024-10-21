# WebcamEyetracking

## Objectives
The project aims to develop a computer vision module of a virtual assistant. The operation of the system is related to **two macro functionalities**:
- The first functionality is the **recognition of the user who is using the system**, based on his face. In case he is not recognized, he will have to register.
- The second is the assistant's detection that the **user is looking at the webcam** of the device. If so, the system will activate and make itself available to the user.

## Project structure
The files that make up the project are as follows:
- ***main.py***: main file of the program, which contains the main() function and all the variables and functions related to the graphical appearance, such as colors, text and fonts.
- ***logic.py***: file containing the program logic.
- ***imageRecognizer.py***: file defining the class of the same name, which is responsible for recognizing the user's face.
- ***registered_user.json***: file where registered user data is stored.
- ***predictor/shape_predictor_68_face_landmarks.dat***: predictor used for the recognition of 68 face landmarks.
- ***Predictor/faceLandmarks.jpg***: display of the coordinates of the 68 points of facial landmarks.

## Design and implementation choices
### Face recognition
First, the system will detect, through the use of the webcam, all the faces in each single frame, going to highlight them through the use of bounding boxes. The assistant, in the case there are several people, will consider only the closest one and its bounding box will be green in color.
Specifically, to do this, we implemented the detect_faces(...) function, which which takes as input the frame and, through the use of the function *get_frontal_face_detector(...)* of the dlib library (http://dlib.net/), returns the faces detected, placing the closest one first.
![Screenshot 2024-10-21 175142](https://github.com/user-attachments/assets/d255526b-bdf5-4567-86a8-84385e4e78f2)

Regarding face recognition functionality, a class, called **ImageRecognizer**, has been defined, which encapsulates the data structures and algorithms to perform this task. The class is based on the face_recognition library (https://github.com/ageitgey/face_recognition), which provides functions for the face recognition based in turn on the dlib API.
The functions in this library use a one-shot neural network, i.e., one that is able to recognize a person's face by having only one reference image of the man or woman to be recognized. The neural network is used to define an array of values, each of which corresponds to a characteristic (feature) of the face, and only then a quadratic difference algorithm is used to define how much the two faces resemble each other. The more the result obtained from the procedure just described is close to zero, the more the two compared faces resemble each other. In concrete terms, face_recognition defines the following features:
- **face_encodings(...)**: which takes as input the image containing the face and its bounding box, returning the encoding of the face, i.e., the feature vector. 
- **compare_faces(...)**: which needs as input a list of encoded faces and the face, also encoded, to be compared; what it returns is an array in which each value can take the value 0 or 1, identifying whether or not the face to be compared resembles the vector of faces passed as the first parameter.
- **face_distance(...)**: is similar to the previous function. In this case, however an array is returned whose values define the Euclidean distance of a face from the one to be recognized.
To recognize the face of the person who is using the service, the last two functions are called. Specifically, ***face_distance(...)*** is used first, which allows us to figure out which face, among the registered ones, most resembles the user. Next, via ***compare_faces(...)*** we check whether or not the similar face matches the that of the user. In case of a negative outcome, the user will be asked to register, by entering their name from the console. In order not to block the execution of the camera, the console is handled with a separate thread, all in a secure manner.
The faces and names of registered people are also stored on disk, in an encoded, in a JSON-type file. The data are stored in an encoded manner, so as to optimize and speed up their reading and writing.
Specifically, the file is read only when an object of type ImageRecognizer (thus each time the program is started) and modified each time a a new user is registered. In fact, there are two lists in the ImageRecognizer class. where the read values of faces and names are saved, respectively.

### View tracking
After the face recognition phase, the function is called ***face_landmark_detector(...)***, which takes as input the frame and the detected face and, through the use of the ***shape_predictor(...)*** function of the dlib library and the file *shape_predictor_68_face_landmarks.dat*, returns the landmarks of the the latter. So, it turns out that it is possible to distinguish the reference points of individual eyes (see as a reference for values the image *faceLandmarks.jpg*).
At this point, via the function ***is_looking_at_cam(...)***, which takes as input the frame and the reference points of an eye, we check whether or not that eye is looking at the webcam.
Specifically, the function first converts the image to grayscale and, based on the coordinates of the landmarks, goes on to extrapolate from the image the portion of the face related to the analyzed eye (A). Next, an averaging filter (B) and a Gaussian filter (C) are applied consecutively, with the aim of “cleaning up” the image of possible noise, such as veins, eyelashes and noise. Then a dynamic binary thresholding operation is performed, using the Otsu technique (D), with the purpose of distinguishing the pupil from the rest. 
Since this is not sufficient for good pupil isolation, an aperture (erosion followed by dilation) of the image (E) is made, with the aim of going to try to eliminate what the pupil does not have to do. However, the image may still have noise and, in addition, the pupil region may contain holes in it, caused, for example, by the reflection of light. To solve the latter problems, the following ***__largest_component_mask(...)*** function is used, which detects the largest connected region, with connectivity 8, in the image, performs a filling operation on it, and cleans the image of everything that is not part of it (F). 
At this point the function, based on the position of the pupil within the image, determines whether or not the eye is in the center, and consequently whether or not it is looking at the webcam.
For the program, the user is looking at the webcam only in the case where both eye analysis responses are simultaneously affirmative.
![Screenshot 2024-10-21 180211](https://github.com/user-attachments/assets/453bf44d-45ef-44aa-9298-e14b96cf8bcb)
