import cv2
import csv

import numpy as np

from utils.label_converters import label_to_int
from sklearn.tree import DecisionTreeClassifier

# Agarro las cosas en los archivos las guardo en variables y las mando a train data y labels
def load_training_set():
    train_data = []
    train_labels = []
    with open('generated-files/shapes-hu-moments.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            class_label = row.pop() # saca el ultimo elemento de la lista
            floats = []
            for n in row:
                floats.append(float(n)) # tiene los momentos de Hu transformados a float.
            train_data.append(np.array(floats, dtype=np.float32)) # momentos de Hu
            train_labels.append(np.array([label_to_int(class_label)], dtype=np.int32)) # Resultados
            #Valores y resultados se necesitan por separados
    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)
    return train_data, train_labels
# transforma los arrays a arrays de forma numpy

# llama la funcion de arriba, se manda a entrenar y devuelve el modelo entrenado
def train_model():
    train_data, train_labels = load_training_set()

    tree = DecisionTreeClassifier(max_depth=10)
    tree.fit(train_data, train_labels.ravel())
    return tree

