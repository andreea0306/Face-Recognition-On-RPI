import os
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# preprocess_image prepares image for feeding it into MobileNetV3 model
def preprocess_image(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # MobileNetV3 expects input in the range [-1, 1]
    img_array = preprocess_input(img_array)
    return img_array

# MobileNetV3Small model without the top classification layer
base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# extract features using MobileNetV3
def extract_features(face):
    face_array = preprocess_image(face)
    features = base_model.predict(face_array)
    features = np.reshape(features, (features.shape[0], -1))
    return features

def detect_name(predicted_label):
    # mapping between labels and person names
    label_to_person = {
        0: "borisov",
        1: "dimitar_barisov",
        2: "lilia",
        3: "nina_dobrev",
        4: "rumen_radev"
    }

    # return the corresponding person name for the predicted label
    return label_to_person.get(predicted_label, "Unknown")

# detect faces in an image and return the cropped face
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load the pre-trained face detector model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Get the first detected face (assuming there is only one face in the image)
        x, y, w, h = faces[0]
        # crop the face from image
        face = img[y:y + h, x:x + w]
        # resize face to the required input size for MobileNetV3
        face = cv2.resize(face, (224, 224))
        return face
    else:
        return None

# display image with the detected face
def display_face(image, detected_face):
    plt.figure(figsize=(8, 8))

    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    # Display the detected face
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB))
    plt.title("Detected Face")
    plt.axis("off")

    plt.show()
    
# load dataset
dataset_path = "./"

# Create embeddings for all images in the reference path only
embeddings = []
labels = []
for label, personality in enumerate(sorted(os.listdir(os.path.join(dataset_path, "train")))):
    personality_path = os.path.join(dataset_path, "train", personality)
    for image_file in os.listdir(personality_path):
        image_path = os.path.join(personality_path, image_file)
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        # get the cropped face
        face = detect_face(img)
        
        if face is not None:
	    #debug - display face cropp
            #display_face(img, detect_face(img))
            # extract features
            face_embedding = extract_features(face)
            embeddings.append(face_embedding.flatten())  # Flatten the features
            labels.append(label)

# convert lists to NumPy arrays to use them to train the k-NN classifier
embeddings = np.array(embeddings)
labels = np.array(labels)

# train a simple k-NN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(embeddings, labels)

# face recognition
def predicted_face(test_img):
    # get the cropped face
    face = detect_face(test_img)

    if face is not None:
        # extract features
        test_embedding = extract_features(face).flatten()  # Flatten the features
        # find the 3 closest neighbors
        distances, indices = knn_classifier.kneighbors(test_embedding.reshape(1, -1), n_neighbors=3)
        predicted_labels = knn_classifier.predict(test_embedding.reshape(1, -1))
        return predicted_labels, indices.flatten(), distances.flatten()
    else:
        return None, None, None

actual_labels_list = []
predicted_labels_list = []
distances_list = []
indices_list = []

# test the face recognition system for every image in the "test" directory
for label, personality in enumerate(sorted(os.listdir(os.path.join(dataset_path, "test")))):
    test_personality_path = os.path.join(dataset_path, "test", personality)
    for image_file in os.listdir(test_personality_path):
        test_image_path = os.path.join(test_personality_path, image_file)

        test_img = cv2.imread(test_image_path)
        # predicted_face function - get the predicted label, indices, and distances
        predicted_labels, indices, distances = predicted_face(test_img)

        if predicted_labels is not None:
            person_name = detect_name(predicted_labels[0])
            reference_names = [detect_name(labels[i]) for i in indices]

            print("Input Image:", test_image_path)
            print("Predicted Person:", person_name)
           # print("Closest References:", reference_names)
           # print("Distances:", distances[:3])
            print()

	    #display img - debug
            #display_face(test_img, detect_face(test_img))

            # append actual and predicted labels, distances, and indices
            actual_labels_list.append(label)
            predicted_labels_list.append(predicted_labels[0])
            distances_list.append(distances[:3])
            indices_list.append(indices)

# convert lists to NumPy arrays to compute the accuracy
actual_labels_array = np.array(actual_labels_list)
predicted_labels_array = np.array(predicted_labels_list)

# compute accuracy
accuracy = accuracy_score(actual_labels_array, predicted_labels_array)
print("Accuracy:", accuracy)

# display the first 3 distances and their corresponding names for each prediction
for i in range(len(actual_labels_list)):
    print(f"Image {i+1} - Actual: {detect_name(actual_labels_list[i])}, Predicted: {detect_name(predicted_labels_list[i])}")
    print("Closest References:", [detect_name(labels[j]) for j in indices_list[i]])
    print("Distances:", distances_list[i])
    print()
