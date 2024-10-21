import json
import face_recognition
import cv2
import numpy as np
import threading

# Class that tries to recognize the user's face from a frame in BGR encoding
class ImageRecognizer:
    # Fields and methods of the class
    __known_faces = []  # Array of known faces
    __known_names = []  # Array of known names
    __file_path = "registered_user.json"    # JSON file path
    __name_string = "Write your name: "     # String for user's name 
    __username = ""     # User's name
    __data_lock = threading.Lock()     # Data lock

    # Builder method
    def __init__(self):
        self.__known_faces, self.__known_names = self.__generate_images_array()
 
    def __generate_images_array(self):
        """
        Loads known face encodings and names from a JSON file.
        :return: A tuple containing the list of known face encodings and the list of known names.
        """
        face_encodings = []
        face_names = []
        
        with open(self.__file_path, 'r') as file:
            users = json.load(file)
        
        for user in users:
            face_encodings.append(np.array(user["encoding"]))
            face_names.append(user["name"])
        
        return face_encodings, face_names

    def __save_face(self, face_encoding, name_id):
        """
        Saves the face encoding and name in a JSON file for all registered users.

        :param face_encoding: encoding of the user's face
        :param name_id: ID or name of the user
        """
        res = []
        json_file = open(self.__file_path, "w")

        # Append the new face encoding and name to the known faces list
        self.__known_faces.append(face_encoding[0])
        self.__known_names.append(name_id)

        # Create the JSON representation for each user
        for user in zip(np.arange(len(self.__known_names), dtype=int), self.__known_names, self.__known_faces):
            res.append({"Id": int(user[0]), "name": user[1], "encoding": user[2].tolist()})

        # Write the JSON data to the file
        json.dump(res, json_file)
        json_file.close()

    def __sign_in(self, face_encoding):
        """
        Signs in a user, asking for its name, and saving the face encoding and name in a JSON file.

        :param face_encoding: encoding of the user's face
        """
        with self.__data_lock:
            # Ask for the user's name
            self.__username = input(self.__name_string)

            # Save the face encoding and name in a JSON file for all registered users
            self.__save_face(face_encoding, self.__username)

        print("The user has been registered!")

    def __thread_sign_in(self, face_encoding):
        """
        Starts a new thread that will sign in a user, asking for its name, and saving the face encoding and name in a JSON file.
        :param face_encoding: encoding of the user's face
        """
        # Create a new thread that will sign in the user
        thread = threading.Thread(target=self.__sign_in, args=(face_encoding, ))
        # Start the thread
        thread.start()

    def recognize_face(self, frame, face_bounding_box):
        """
        Tries to recognize the user's face.
        :param frame: frame containing the user's face in BGR encoding.
        :param face_bounding_box: bounding box of the user's face (x, y, w, h).
        :return: the user's name if recognized, otherwise starts a thread to sign in the user.
        """
        # Convert frame to RGB
        small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get face encoding
        face_encoding = face_recognition.face_encodings(small_frame, list([face_bounding_box]))

        # Check if data lock is active
        if self.__data_lock.locked():
            return ""

        # Check if known faces exist
        if len(self.__known_faces) != 0:
            # Compare face encodings
            matches = face_recognition.compare_faces(self.__known_faces, face_encoding[0])
            face_distances = face_recognition.face_distance(self.__known_faces, face_encoding[0])
            best_match_index = np.argmin(face_distances)
            
            # Check if a match is found
            if matches[best_match_index]:
                self.__username = self.__known_names[best_match_index]
                return self.__username

        # User is not recognized, start a thread for sign-in
        self.__thread_sign_in(face_encoding)

        return ""
