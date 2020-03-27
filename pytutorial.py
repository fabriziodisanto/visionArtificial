import cv2
import numpy as np
# e1 = cv2.getTickCount()
# start = datetime.now()
#
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, 1)

    # Display the resulting frame
    cv2.imshow('frame', cv2.flip(frame, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
# end = datetime.now()
#
#
# # your code execution
# e2 = cv2.getTickCount()
# time = (e2 - e1)/ cv2.getTickFrequency()
# print(time)
# time = end - start
# print(time)

#
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.mov', fourcc, 20.0, (640,480))
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         frame = cv2.flip(frame, 0)
#
#         # write the flipped frame
#         out.write(frame)
#
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
#
# # Release everything if job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()

#
# while(True):
#     ret, frame = cap.read()  # return a single frame in variable `frame`
#     norm = cv2.flip(frame, 1)
#     cv2.imshow('img1',frame) #display the captured image
#     if cv2.waitKey(1) == ord('y'): #save on pressing 'y'
#         cv2.imwrite('c1.png',frame)
#         break
#
# cv2.destroyAllWindows()
# cap.release()


# cam = cv2.VideoCapture(0)
#
# cv2.namedWindow("test")
#
#
# while True:
#     ret, frame = cam.read()
#     cv2.imshow("test", frame)
#     if not ret:
#         break
#     k = cv2.waitKey(1)
#
#     # if k%256 == 27:
#     if k == ord('y'):
#         # y pressed
#         print("Closing...")
#         break
#     elif k == ord(' '):
#         # SPACE pressed
#         img_name = "opencv_frame_{}.png".format(str(datetime.now()))
#         cv2.imwrite(img_name, frame)
#         print("{} written!".format(img_name))
#
# cam.release()
#
# cv2.destroyAllWindows()