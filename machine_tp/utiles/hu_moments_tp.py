import cv2
import numpy as np
import math
import glob
import csv


# Escribo los valores de los momentos de Hu en el archivo
def write_hu_moments(label, writer):
    files = glob.glob('./shapes.tp/' + label + '/*')  # label recibe el nombre de la carpeta
    hu_moments = []
    for file in files:
        hu_moments.append(hu_moments_of_file(file))
    for mom in hu_moments:
        flattened = mom.ravel()  # paso de un array de arrays a un array simple.
        row = np.append(flattened, label)  # le metes el flattened array y le agregas el label
        writer.writerow(row)  # Escribe una linea en el archivo.


def generate_hu_moments_file():
    with open('archivo_hu_tp/figuras_hu.csv', 'w',
              newline='') as file:  # Se genera un archivo nuevo (W=Write)
        writer = csv.writer(file)
        # Ahora escribo los momentos de Hu de cada uno de las figuras. Con el string "rectangle...etc" busca en la carpeta donde estan cada una de las imagenes
        # generar los momentos de Hu y los escribe sobre este archivo. (LOS DE ENTRENAMIENTO).
        write_hu_moments("estrella_tp", writer)
        write_hu_moments("rectangulo_tp", writer)
        write_hu_moments("triangulo_tp", writer)



def hu_moments_of_file(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 67, 2)

    # Invert the image so the area of the shape is filled with 1's
    bin = 255 - bin

    # TÉCNICAS DE LIMPIEZA MEJORADAS
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    
    # 1. Opening para eliminar ruido pequeño
    bin = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel_small)
    
    # 2. Closing para rellenar pequeños agujeros
    bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel_small)
    
    # 3. Filtro de mediana para eliminar ruido sal y pimienta
    bin = cv2.medianBlur(bin, 5)
    
    # 4. Encontrar contornos y quedarse solo con el más grande
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Si no se encuentran contornos, devolver momentos de Hu nulos
        return np.zeros((7, 1))
    
    # Encontrar el contorno con mayor área
    shape_contour = max(contours, key=cv2.contourArea)
    
    # 5. Crear imagen limpia solo con el objeto principal
    clean_image = np.zeros_like(bin)
    cv2.fillPoly(clean_image, [shape_contour], 255)
    
    # 6. Suavizado final
    clean_image = cv2.morphologyEx(clean_image, cv2.MORPH_CLOSE, kernel_medium)
    
    # Recalcular contornos en la imagen limpia
    contours, _ = cv2.findContours(clean_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_contour = max(contours, key=cv2.contourArea)

    # Calculate Moments
    moments = cv2.moments(shape_contour)
    
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    
    # Log scale hu moments
    for i in range(0, 7):
        if huMoments[i] != 0:  # Evitar log(0)
            huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
        else:
            huMoments[i] = 0
            
    return huMoments