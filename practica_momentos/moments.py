from math import sqrt, copysign, log10

import cv2 as cv

# Ejercicio 1
# Obtener el contorno de una figura principal, dibujarlo en la imagen
# y dibujar también una marca en el centroide y una circunferencia
# de radio proporcional a la raíz cuadrada de m00

def exercise_one():
    image = cv.imread('../static/images/phone.png')
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    cv.imshow("original", image)

    contours, hierarchy = cv.findContours(thresh1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnt = contours[1]
    # compute the center of the contour
    M = cv.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # draw the contour and center of the shape on the image
    cv.drawContours(image, [cnt], -1, (0, 0, 255), 2)
    cv.circle(image, (cX, cY), int(sqrt(M["m00"])), (255, 0, 0), 2)
    cv.circle(image, (cX, cY), 3, (255, 0, 0), -1)
    cv.putText(image, "center", (cX - 20, cY - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv.imshow("Image", image)
    cv.waitKey(0)

# Ejercicio 3
# Ídem anterior imprimiendo los invariantes de Hu en lugar de los momentos
# Para obtener la matriz de Hu usen el siguiente metodo:
#     huMoments = cv.HuMoments(M)

def exercise_three():
    letter_s_one = cv.imread('../static/images/letterSOne.png')
    gray1 = cv.cvtColor(letter_s_one, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray1, 127, 255, cv.THRESH_BINARY)
    cv.imshow("letter_s_one", letter_s_one)

    contours1, hierarchy = cv.findContours(thresh1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnt1 = contours1[1]
    # compute the center of the contour
    cv.drawContours(letter_s_one, [cnt1], -1, (0, 0, 255), 2)
    cv.imshow("contorno letter s one", letter_s_one)
    moments1 = cv.moments(cnt1)
    huMoments1 = cv.HuMoments(moments1)
    # hu[0] is not comparable in magnitude as hu[6].
    # We can use use a log transform given below to bring them in the same range
    for i in range(0, 7):
        huMoments1[i] = -1 * copysign(1.0, huMoments1[i]) * log10(abs(huMoments1[i]))
        print("letter s1 -> h" + str(i) + " " + str(huMoments1[i]))

    letter_s_two = cv.imread('../static/images/letterSTwo.png')
    gray2 = cv.cvtColor(letter_s_two, cv.COLOR_RGB2GRAY)
    ret2, thresh2 = cv.threshold(gray2, 127, 255, cv.THRESH_BINARY)
    cv.imshow("letter_s_two", letter_s_two)

    contours2, hierarchy = cv.findContours(thresh2, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnt2 = contours2[1]
    # compute the center of the contour
    cv.drawContours(letter_s_two, [cnt2], -1, (0, 0, 255), 2)
    cv.imshow("contorno letter s two", letter_s_two)
    moments2 = cv.moments(cnt2)
    huMoments2 = cv.HuMoments(moments2)
    # hu[0] is not comparable in magnitude as hu[6].
    # We can use use a log transform given below to bring them in the same range
    for i in range(0, 7):
        huMoments2[i] = -1 * copysign(1.0, huMoments2[i]) * log10(abs(huMoments2[i]))
        print("letter s2 -> h" + str(i) + " " + str(huMoments2[i]))

    cv.waitKey(0)

# Ejercicio 4
# Obtener los invariantes de Hu para una forma, y reconocerla luego en otras imágenes

def exercise_four():
    letter_s = cv.imread('../static/images/letterSTwo.png')
    gray1 = cv.cvtColor(letter_s, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray1, 127, 255, cv.THRESH_BINARY)
    contours1, hierarchy = cv.findContours(thresh1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnt1 = contours1[1]
    cv.drawContours(letter_s, [cnt1], -1, (0, 0, 255), 2)
    cv.imshow("letter_s_one", letter_s)
    moments = cv.moments(cnt1)
    cv.waitKey(0)
    huMoments = cv.HuMoments(moments)
#     now in huMoments1 we have letter s invariants
#     lets use shape match
    alphabet = cv.imread('../static/images/alphabet.jpg')
    gray2 = cv.cvtColor(alphabet, cv.COLOR_RGB2GRAY)
    ret2, thresh2 = cv.threshold(gray2, 127, 255, cv.THRESH_BINARY)
    cv.imshow("alphabet", alphabet)
    contours_alphabet, hierarchy = cv.findContours(thresh2, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contour in contours_alphabet:
        moments_alphabet = cv.moments(contour)
        huMoments_alphabet = cv.HuMoments(moments_alphabet)
        if cv.matchShapes(contour, cnt1, cv.CONTOURS_MATCH_I2, 0) < 0.4:
            cv.drawContours(alphabet, contour, -1, (0, 0, 255), 2)
            cv.imshow("contornos", alphabet)
            cv.waitKey(0)


# exercise_one()
# exercise_three()
exercise_four()