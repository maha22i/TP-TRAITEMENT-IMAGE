# Partie 2 : Exercice 3

import cv2
import numpy as np

# Charger le modèle de détection des visages avec OpenCV
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')

# Charger les modèles de genre et d'âge (en utilisant des modèles pré-entraînés Caffe)
age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')

# Classes d'âge et de genre
age_classes = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
gender_classes = ['Homme', 'Femme']

# Ouvrir une vidéo ou une image
cap = cv2.VideoCapture(0)  # Utiliser 0 pour la caméra par défaut

while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    # Préparation de l'image pour le réseau
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Seuil de confiance
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extraire la région du visage
            face = frame[startY:endY, startX:endX]
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # Prédire le genre
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = gender_classes[gender_preds[0].argmax()]

            # Prédire l'âge
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = age_classes[age_preds[0].argmax()]

            # Dessiner le rectangle autour du visage et ajouter le texte
            label = f"{gender}, {age}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Afficher le résultat
    cv2.imshow("Détection des visages", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
