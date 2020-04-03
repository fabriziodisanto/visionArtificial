import cv2 as cv

# ejercicio 2
# Aplicar los métodos de umbral automático
image = cv.imread('../static/images/messi.jpg')
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
# primer metodo OTSU
ret1, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("OTSU", thresh1)
# segundo metodo TRIANGLE
ret2, thresh2 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)
cv.imshow("TRIANGLE", thresh2)
cv.waitKey()