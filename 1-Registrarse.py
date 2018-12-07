import cv2
import numpy as np
import os

nomb = input("Introduzca su nombre: ")
DirectoryPath = 'Database/'+str(nomb)
os.mkdir(DirectoryPath)

input("Presione enter para generar su carpeta de datos")

cam = cv2.VideoCapture(0)

cascPath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

contador = 0

while(True):
    _, imagen_marco = cam.read()

    grises = cv2.cvtColor(imagen_marco, cv2.COLOR_BGR2GRAY)

    rostro = faceCascade.detectMultiScale(grises, 1.5, 5)

    for (x,y,w,h) in rostro:
        cv2.rectangle(imagen_marco, (x,y), (x+w, y+h), (255,0,0), 4)
        contador += 1

        cv2.imwrite("Database/"+nomb+"/"+nomb+"_"+str(contador)+".jpg", grises[y:y+h, x:x+w])
        cv2.imshow("Registrando tu perfil en la base de datos ...", imagen_marco)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
    
    elif contador >= 100:
        break

cam.release()
cv2.destroyAllWindows()

        
