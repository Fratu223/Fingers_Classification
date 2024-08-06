from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import load_img, img_to_array
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import shutil
import cv2
import os

def creating_classes(folder_name:str, classes:int):
    """_summary_

    Args:
        folder_name (str): _description_
        classes (int): _description_
    """


    image_paths = []
    for image_name in os.listdir(folder_name):
        image_paths.append(os.path.join(folder_name, image_name))
    
    model = VGG16(weights='imagenet', include_top=False)

    def extract_features(img_path, model):
        img = load_img(img_path, target_size=(224, 224))
        img_data = img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = model.predict(img_data)
        return features.flatten()
    
    features = [extract_features(img_path, model) for img_path in image_paths]

    pca = PCA(n_components=50)
    reduced_features = pca.fit_transform(features)
    
    kmeans = KMeans(n_clusters=classes) 
    clusters = kmeans.fit_predict(reduced_features)

    i = 0

    while i < classes:
        os.makedirs(f'cluster_{i}', exist_ok=True)
        i+=1

    for img_path, cluster in zip(image_paths, clusters):
        shutil.move(img_path, os.path.join(f'cluster_{cluster}', os.path.basename(img_path)))


def load_data(folders):
    """_summary_

    Args:
        folders (_type_): _description_

    Returns:
        _type_: _description_
    """


    images = []
    labels = []
    for label, folder in enumerate(folders):
        folder_path = folder
        for file in os.listdir(folder_path):
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(folder_path, file)
                img = load_img(img_path, target_size=(64, 64))  # Resize images to 64x64
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_image(frame):
    """_summary_

    Args:
        frame (_type_): _description_

    Returns:
        _type_: _description_
    """


    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img