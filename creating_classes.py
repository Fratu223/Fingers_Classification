from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import load_img, img_to_array
from keras.preprocessing import image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os

def creating_classes(folder_name:str, classes:int):
    image_paths = []
    for image_name in os.listdir(folder_name):
        image_paths.append(os.path.join(folder_name, image_name))
    
    model = VGG16(weights='imagenet', include_top=False)

    features = []

    for img_path in image_paths:
        img = load_img(img_path, target_size=(224, 224))
        img_data = img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        feature = model.predict(img_data)
        features.append(feature)

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