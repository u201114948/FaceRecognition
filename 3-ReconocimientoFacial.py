import cv2
import pickle
import os
import winsound
import time


cascPath = "Cascades/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

eyeCascade = cv2.CascadeClassifier("Cascades/haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier("Cascades/haarcascade_smile.xml")

reconocimiento = cv2.face.LBPHFaceRecognizer_create()
reconocimiento.read("nuevosdatos.yml")

etiquetas = { "nombre_persona" : 1 }

with open("labels.pickle", 'rb') as f:
    pre_etiquetas = pickle.load(f)
    etiquetas = { v:k for k,v in pre_etiquetas.items() }
	
web_cam = cv2.VideoCapture(0)

count = 0

while(True):
    ret, marco = web_cam.read()
    grises = cv2.cvtColor(marco, cv2.COLOR_BGR2GRAY)
    rostros = faceCascade.detectMultiScale(grises, 1.5, 5)

    for (x, y, w, h) in rostros:
        #print(x,y,w,h)
        roi_gray = grises[y:y+h, x:x+w]
        roi_color = marco[y:y+h, x:x+w]

        # reconocimiento Facial
        id_, conf = reconocimiento.predict(roi_gray)
        #if conf >= 10 and conf < 48:
        print(conf)
        if conf >= 10 and conf < 28:
            #print(id_)
            #print(etiquetas[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            count += 1

            nombre = etiquetas[id_]

            if conf > 50:
                #print(conf)
                nombre = "Comparando..."

            color = (255,255,255)
            grosor = 2
            cv2.putText(marco, nombre, (x,y), font, 1, color, grosor, cv2.LINE_AA)

        cv2.rectangle(marco, (x,y), (x+w, y+h), (0,255,0), 2)

        rasgos = smileCascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in rasgos:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255,0), 2)

   # Display resize del marco
    marco_display = cv2.resize(marco, (1200, 650), interpolation = cv2.INTER_CUBIC)
    cv2.imshow('Detectando Rostros - Presione (e) para salir', marco_display)

    if count >= 8:
        count = True
        break

    elif cv2.waitKey(1) & 0xFF == ord('e'):
        break


# Si el usuario es reconocido sonar acceso
if count == True:
    winsound.PlaySound("access_granted.wav", winsound.SND_FILENAME)
    time.sleep(1)
                
# Cuando todo esta hecho, liberamos la captura
web_cam.release()
cv2.destroyAllWindows()

input("Presione enter para salir")                        
                
