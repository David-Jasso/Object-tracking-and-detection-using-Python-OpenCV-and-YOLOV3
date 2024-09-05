import cv2 as cv
import numpy as np
from random import randint
from yolo import yolo_predict, yolo_postprocess

# Inicializar parametros para YOLOV3.
objectnessThreshold = 0.5 # Umbral de objetualidad (0-1).
confThreshold = 0.5       # Umbral de confianza (0-1).
nmsThreshold = 0.4        # Umbral de supresión no máxima NMS.
inpWidth = 416            # Anchura de la imagen de entrada.
inpHeight = 416           # Altura de la imagen de entrada.

# Cargar las clase que contiene todos los objetos para entrenar el modelo.
classesFile = "ProyectoFinal/models/coco.names"
classes = []

# Abrir el archivo en modo lectura y almacenar las classes en una lista.
with open(classesFile, 'rt') as f: 
    classes = f.read().rstrip('\n').split('\n')

# Cargar configuración y pesos de la red neuronal.
modelConfig = "ProyectoFinal/models/yolov3.cfg"
modelWeights = "ProyectoFinal/models/yolov3.weights"

# Cargar el modelo a partir de los pesos y la configuración dada.
yolo = cv.dnn.readNetFromDarknet(modelConfig, modelWeights)

# Iniciar MultiTracker para seguimiento de objetos
trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

trackerType = "CSRT"

# Crear bandera para habilitar el seguimiento.
tracking_enabled = False

# Variable para almacenar el valor numerico de las teclas del 1 al 9.
p = 0

# Iniciar captura de video.
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error al conectar la cámara")
    exit()

# Bucle infinito.
while True:
    ret, frame = cap.read()
    key = cv.waitKey(1)

    if not ret: 
        print("Error reading frame")
        break

    if key == ord('q'): break
    elif key == ord('d'):
        # Procesamiento de la red neuronal.
        outs = yolo_predict(frame, inpHeight, inpWidth, yolo)
        print(outs[0])
        
        # Eliminar los recuadros delimitadores de baja confianza.
        indices, boxes, confidence, classId = yolo_postprocess(frame, outs, objectnessThreshold, confThreshold, nmsThreshold, classes)
        print(classId) # Impresion de las clases de los objetos detectados.
        print(indices) # Impresion de los indices de los objetos detectados.
        print(boxes)   # Impresion de los recuadros de los objetos detectados.

        # Ordenar clases.
        classOrd = classId
        classOrd.sort()
        print(classOrd)

        # Obtener tamaño del arreglo de las clases ordenadas.
        l = len(classOrd)
        print(l)

        posicion = []
        i = 0

        # Crear arreglo que almacene solo las clases detectadas.
        while i < l-1:
            print(classOrd[i])
            if classOrd[i] != classOrd[i+1]:
                    posicion.append(classOrd[i])
            i+=1
        posicion.append(classOrd[l-1])
        
        print(posicion)

        # Mostrar Deteccion de Objetos.
        cv.imshow("Video", frame)
        cv.waitKey(0)

    # Seleccionar clase para seguimiento con las teclas del 1 al 9.
    elif key >= ord('1') and key <= ord('9'):
        TP = int(key)
        k = 1
        i = 49
        p = 0
        while k:
            if TP == i:
                k  = 0
            p += 1
            i += 1
        print(p)
    # Realizar el seguimiento de los objetos a partir de la clase seleccionada. 
    elif key == ord('t') and p <= len(posicion):
        multiTracker = cv.legacy.MultiTracker_create()
        for i, bbox in enumerate(boxes):
            if classId[i] == posicion[p-1]:
                multiTracker.add(cv.legacy.TrackerCSRT_create(), frame, bbox)
                print(bbox)
        
        # Habilitar bandera para realizar el seguimiento.
        tracking_enabled = True

    elif tracking_enabled:
        # Mostrar seguimiento de Objetos.
        success, bboxes = multiTracker.update(frame)

        if success:
            for i, newbox in enumerate(bboxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv.rectangle(frame, p1, p2, (255, 0, 0), 2, cv.LINE_AA)
        
    cv.imshow("Video", frame)
    
#Destruir las ventanas creadas para liberar espacio
cap.release()
cv.destroyAllWindows()
