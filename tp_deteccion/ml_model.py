import cv2
import csv

import numpy as np
from sklearn.tree import DecisionTreeClassifier


def label_to_int(string_label):
    if string_label == '5-point-star': return 1
    if string_label == 'rectangle': return 2
    if string_label == 'triangle':
        return 3

    else:
        raise Exception('unkown class_label')


def int_to_label(string_label):
    if string_label == 1: return '5-point-star'
    if string_label == 2: return 'rectangle'
    if string_label == 3:
        return 'triangle'
    else:
        raise Exception('unkown class_label')


trainData = []
trainLabels = []

# Agarro las cosas en los archivos las guardo en variables y las mando a train data y labels
def load_training_set():
    global trainData
    global trainLabels
    with open('./machine/generated-files/shapes-hu-moments.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            class_label = row.pop() # saca el ultimo elemento de la lista
            floats = []
            for n in row:
                floats.append(float(n)) # tiene los momentos de Hu transformados a float.
            trainData.append(np.array(floats, dtype=np.float32)) # momentos de Hu
            trainLabels.append(np.array([label_to_int(class_label)], dtype=np.int32)) # Resultados
            #Valores y resultados se necesitan por separados
    trainData = np.array(trainData, dtype=np.float32)
    trainLabels = np.array(trainLabels, dtype=np.int32)
# transforma los arrays a arrays de forma numpy


# llama la funcion de arriba, se manda a entrenar y devuelve el modelo entrenado
def train_model():
    load_training_set()

    tree = DecisionTreeClassifier(max_depth=10)
    tree.fit(trainData, trainLabels.ravel())
    return tree
