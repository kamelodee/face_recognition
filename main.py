import  numpy as np
import face_recognition
import cv2

img = face_recognition.load_image_file('images/bill2.jpg')
img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

img_test = face_recognition.load_image_file('images/bill3.jpg')
img_test = cv2.cvtColor(img_test,cv2.COLOR_RGB2BGR)

faceloc = face_recognition.face_locations(img)[0]

faceloctest = face_recognition.face_locations(img_test)[0]

cv2.rectangle(img,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)


cv2.rectangle(img_test,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)


encodeimg = face_recognition.face_encodings(img)[0]
encodeimgtest = face_recognition.face_encodings(img_test)[0]

match = face_recognition.compare_faces([encodeimg],encodeimgtest)

faceDis = face_recognition.face_distance([encodeimg],encodeimgtest)
cv2.putText(img_test,f'{match} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

print(match,faceDis)
print(faceloc,faceloctest)

cv2.imshow("bill gates",img)
cv2.imshow("bill gates test",img_test)

cv2.waitKey(0)