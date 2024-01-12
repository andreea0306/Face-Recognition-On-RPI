import os
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def display_personality_images(dataset_path, subset="train", save_path=None):
    plt.figure(figsize=(15, 10))

    for label, personality in enumerate(sorted(os.listdir(os.path.join(dataset_path, subset)))):
        personality_path = os.path.join(dataset_path, subset, personality)
        image_files = os.listdir(personality_path)
        for i in range(min(7, len(image_files))):
            image_path = os.path.join(personality_path, image_files[i])
            img = cv2.imread(image_path)
            
            plt.subplot(7, 5, label * 5 + i + 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"{personality}", fontsize=8)
            plt.axis("off")
    if save_path:
        plt.savefig(save_path)
    plt.show()

dataset_path = './'
display_personality_images(dataset_path, subset="train", save_path='./train_figure')

display_personality_images(dataset_path, subset="test", save_path='./test_figure')
