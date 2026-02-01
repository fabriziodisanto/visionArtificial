import cv2
import numpy as np

image = cv2.imread(r"C:\Users\User\Documents\ORNELLA\procesamiento\visionArtificial\machine_tp\shapes.tp\estrella_tp\WhatsApp Image 2025-09-10 at 11.21.23.jpeg")

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 67, 2)

# Invert the image so the area of the UAV is filled with 1's
inv = 255 - bin

kernel_small = np.ones((3, 3), np.uint8)  
kernel_medium = np.ones((5, 5), np.uint8)  

cleaned_opening = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel_small)

cleaned_closing = cv2.morphologyEx(cleaned_opening, cv2.MORPH_CLOSE, kernel_small)

median_filtered = cv2.medianBlur(cleaned_closing, 5)

contours, _ = cv2.findContours(median_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

clean_image = np.zeros_like(median_filtered)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    
    cv2.fillPoly(clean_image, [largest_contour], 255)

final_clean = cv2.morphologyEx(clean_image, cv2.MORPH_CLOSE, kernel_medium)

# Mostrar todas las etapas del proceso
cv2.imshow('1. Original Binarizada', bin)
cv2.imshow('2. Invertida', inv)
cv2.imshow('3. Despues de Opening', cleaned_opening)
cv2.imshow('4. Despues de Closing', cleaned_closing)
cv2.imshow('5. Filtro Mediana', median_filtered)
cv2.imshow('6. Solo Objeto Principal', clean_image)
cv2.imshow('7. Resultado Final', final_clean)

cv2.waitKey(0)
cv2.destroyAllWindows()