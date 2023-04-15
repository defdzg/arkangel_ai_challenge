# AI-Challenge Demo v1.0
## 📄 Description

This Python🐍 application allows to recognize the workers from a company and detect their emotion using a profile picture. It's build on top of the [HSEmotion](https://github.com/HSE-asavchenko/hsemotion) and [Face Recognition](https://github.com/ageitgey/face_recognition) libraries.

A graphical user interface (GUI) is provided to easily use the application, this GUI is built using [Gradio](https://github.com/gradio-app/gradio), a Python🐍 library that allows to easily build web interfaces for machine learning models. This GUI was deployed on [Hugging Face](https://huggingface.co/), a platform that allows to easily deploy and share machine learning models.

The [HSEmotion](https://github.com/HSE-asavchenko/hsemotion) models were pre-trained using [VGGFace2 dataset](https://github.com/ox-vgg/vgg_face2). This application uses the `enet_b2_8` model, wich has an accuracy of `63.03` on the [AffectNet dataset](http://mohammadmahoor.com/affectnet/). The [Face Recognition](https://github.com/ageitgey/face_recognition)  model is built using [dlib](https://github.com/davisking/dlib) and it has a accuracy of `99.38` on the [Labeled Faces in the Wild dataset](http://vis-www.cs.umass.edu/lfw/).

Current demo images were obtained from the [Makers](https://www.makers.build) program. The demo contains 12 images of the fellows from the program. The images filenames contain the name of the fellows.

**⚡ The demo is available [here](https://defdzg-arkangelai-challenge.hf.space/)**.

## 📌 Methodology

The methodology used to build this application is the following:

1. A directory of images is provided to the application, this images are profile pictures of the company's workers.
2. The application obtains the name of the workers from the images' filenames. It also extracts the path of the images.
3. The application uses Face Recognition to detect the faces in the images and obtain the face encodings. Face encodings are 128-dimensional vectors that describe the face.
4. The application uses HSEmotion to detect the emotion of the faces in the images. There are 8 possible emotions: `Anger`, `Contempt`, `Disgust`, `Fear`, `Happiness`, `Neutral`, `Sadness` and `Surprise`.
5. Finally, an unknown image is provided to the application, this image is compared with the face encodings of the workers to determine if the person in the image is a worker of the company. The GUI shows the name of the worker and the emotion detected in the image.

___
Made with ❤️ by Daniel Fernández.