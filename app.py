# These are import statements that import necessary libraries and modules for the app.
import os
import cv2
import gradio as gr
import face_recognition
from hsemotion.facial_emotions import HSEmotionRecognizer

# These lines of code are initializing variables that will be used to store metadata and face
# encodings for workers.
images_path = './workers_images/'
images_in_folder = os.listdir(images_path)
workers_names = []
workers_images_paths = []
workers_faces = []

# These lines of code are initializing an instance of the `HSEmotionRecognizer` class from the
# `facial_emotions` module with a specified `model_name` and `device`. The `model_name` parameter
# specifies the name of the pre-trained deep learning model that will be used for facial emotion
# recognition, and `device` specifies the device on which the model will be run (in this case, the
# CPU).
model_name = 'enet_b2_8'
fer = HSEmotionRecognizer(model_name=model_name, device='cpu')


def get_workers_metadata():
    """
    This function retrieves metadata for workers, including their names and image paths.
    """
    for worker_image in images_in_folder:
        full_name = worker_image.split('.')[0]
        full_name = full_name.replace('_', ' ')
        workers_names.append(full_name)
        worker_image_path = images_path + worker_image
        workers_images_paths.append(worker_image_path)


def get_workers_faces():
    """
    This function reads images of workers, detects their faces, encodes them, and appends the face
    encodings to a list of workers' faces.
    """
    for worker_image_path in workers_images_paths:
        worker_image = cv2.imread(worker_image_path)
        worker_face = face_recognition.face_encodings(worker_image)[0]
        workers_faces.append(worker_face)


def worker_name_recognition(input_image):
    """
    The function recognizes the name of a worker in an input image by comparing their face encoding to a
    list of known workers' face encodings.

    :param input_image: The image of a worker whose name needs to be recognized
    :return: the name of the worker if their face is recognized in the input image, otherwise it returns
    'Unknown'.
    """
    unknown_worker = face_recognition.face_encodings(input_image)[0]
    results = face_recognition.compare_faces(workers_faces, unknown_worker)
    if True in results:
        first_match_index = results.index(True)
        return workers_names[first_match_index]
    else:
        return 'Unknown'


def face_emotion_recognition(input_image):
    """
    The function takes an input image and uses a facial emotion recognition model to predict the emotion
    in the image.

    :param input_image: The input image is the image for which we want to recognize the emotion on the
    face.
    :return: the predicted emotion of a face in the input image.
    """
    emotion, _ = fer.predict_emotions(input_image, logits=True)
    return emotion


def worker_recognition(input_image):
    """
    The function takes an input image, recognizes the worker's name and emotion from the image, and
    returns a string with the worker's name and emotion.

    :param input_image: The input image is the image of a worker's face that is being analyzed for
    worker recognition. The function uses two other functions, face_emotion_recognition and
    worker_name_recognition, to determine the worker's emotion and name respectively.
    """
    emotion = face_emotion_recognition(input_image)
    name = worker_name_recognition(input_image)
    return f'Worker: {name} \nEmotion: {emotion}'


def main():
    """
    This function sets up a Graphical User Interface (GUI) for an the app and launches it, while
    also calling two other functions to retrieve workers' metadata and faces.
    """
    
    examples = ['./demo_images/angry_demo.jpg', './demo_images/disgust_demo.jpg', './demo_images/fear_demo.jpg', './demo_images/Giuliana_demo.jpeg', './demo_images/Jose_demo.jpg']
    
    app = gr.Interface(fn=worker_recognition,
                       inputs="image",
                       outputs="textbox",
                       title='AI-Challenge Demo v1.0',
                       description='This app recognizes workers in images and predicts their emotions.',
                       examples=examples)
    
    get_workers_metadata()
    get_workers_faces()
    # If you want to run the app locally, uncomment the line below and comment out the line after it.
    # app.launch(server_name="0.0.0.0", server_port=80)
    app.launch()


main()
