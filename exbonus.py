# Partie 3 : Bonus

import cv2
import numpy as np

# Fonction pour détecter la peau dans l'image
def detect_skin(frame):
    # Convertir l'image de BGR à YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    # Définir les valeurs limites pour la détection de la peau
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    # Masquer l'image pour détecter les régions correspondant à la peau
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Appliquer quelques filtrages pour améliorer la détection (flou pour réduire le bruit)
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    
    return skin_mask

# Fonction pour reconnaître le geste
def recognize_gesture(contour):
    area = cv2.contourArea(contour)
    
    # Si l'aire est trop petite, on ignore
    if area < 2000:
        return "Aucune main détectée"
    
    # Calculer l'approximation du contour
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Si le nombre de points dans l'approximation est plus élevé, c'est probablement une paume ouverte
    if len(approx) > 5:
        return "Paume ouverte"
    else:
        return "Poing fermé"

# Capture vidéo depuis la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam.")
    exit()

while True:
    # Lire l'image de la caméra
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionner l'image pour un traitement plus rapide
    frame = cv2.resize(frame, (640, 480))

    # Détection de la peau dans l'image
    skin_mask = detect_skin(frame)

    # Trouver les contours dans l'image binaire
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Trouver le plus grand contour (on suppose que c'est la main)
        largest_contour = max(contours, key=cv2.contourArea)

        # Dessiner un rectangle autour de la main
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Reconnaître le geste (paume ou poing)
        gesture = recognize_gesture(largest_contour)
        
        # Afficher le geste détecté à l'écran
        cv2.putText(frame, gesture, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Afficher les images
    cv2.imshow("Détection de la peau", skin_mask)
    cv2.imshow("Reconnaissance des gestes", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
