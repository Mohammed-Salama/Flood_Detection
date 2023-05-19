import os
import cv2
from tensorflow.keras.utils import to_categorical
import imgaug.augmenters as iaa
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
import seaborn as sns


## Global variables

CLASSES = ["flooded", "non-flooded"]
IMG_SIZE = (224, 224)



## Functions

def load_data(data_path):
    data = []
    labels = []
    for c in CLASSES:
        path = os.path.join(data_path, c)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img))
            img_array = cv2.resize(img_array, IMG_SIZE)
            data.append(img_array)
            labels.append(CLASSES.index(c))

    labels = to_categorical(labels)
    return data, labels



def data_augmentation(data, labels):
    augmented_images = []
    augmented_labels = []
    seqs = []
    # Define the augmentation sequences
    for i in range(1):
        seq = iaa.Sequential([
                iaa.Fliplr(p=random.choice([0, 90]) / 360),
                iaa.Flipud(p=random.choice([0, 90]) / 360),
                # iaa.Crop(percent=(0, random.uniform(0, 0.1))),
                iaa.GaussianBlur(sigma=random.uniform(0, 3.0)),
                iaa.AdditiveGaussianNoise(scale=random.uniform(0.01,0.06)*255),
                # iaa.Multiply((random.uniform(0.5, 1.5), random.uniform(0.5, 1.5))),
                # iaa.Affine(
                    # scale={"x": (random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)), "y": (random.uniform(0.8, 1.2), random.uniform(0.8, 1.2))},
                    # translate_percent={"x": (random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)), "y": (random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2))},
                    # rotate=(random.uniform(-45, 45), random.uniform(-45, 45)),
                    # shear=(random.uniform(-16, 16), random.uniform(-16, 16))
                # )
                ], random_order=True)
        seqs.append(seq)

    for i, image in enumerate(data):
        augmented_images.append(image)
        augmented_labels.append(labels[i])
        for seq in seqs:
            augmented_image = seq(image=image)
            augmented_images.append(augmented_image)
            augmented_labels.append(labels[i])
    
    return np.array(augmented_images), np.array(augmented_labels)


def evaluate_model(model, X_test, y_test):
    # predict the test set
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # print the accuracy score
    print("Accuracy: ", accuracy_score(y_test, y_pred))

    # print f1 score
    print("F1 score: ", f1_score(y_test, y_pred, average='macro'))

    # print classification report
    print("Classification report: \n", classification_report(y_test, y_pred))

    # plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()



from pathlib import Path
import imageio

def save_imgs(path:Path, data, labels):
    labels = np.argmax(labels, axis=1)
    for label in np.unique(labels):
        (path/str(label)).mkdir(parents=True,exist_ok=True)
    for i in range(len(data)):
        if(len(labels)!=0):
            imageio.imsave( str( path/str(labels[i])/(str(i)+'.jpg') ), data[i])
        else:
            imageio.imsave( str( path/(str(i)+'.jpg') ), data[i])

