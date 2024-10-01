# Partie 2 : Exercice 4

import cv2

# Initialisation de la webcam
cap = cv2.VideoCapture(0)

# Vérification si la webcam est ouverte correctement
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

# Sélection de l'algorithme de suivi (KCF, CSRT, MIL, etc.)
# Utiliser le sous-module 'cv2.legacy' pour les trackers si nécessaire
tracker = cv2.legacy.TrackerCSRT_create()

# Lire la première image de la vidéo
ret, frame = cap.read()
if not ret:
    print("Erreur lors de la lecture de l'image.")
    cap.release()
    exit()

# Permettre à l'utilisateur de sélectionner la zone de l'objet à suivre
bbox = cv2.selectROI("Sélectionnez l'objet à suivre", frame, fromCenter=False, showCrosshair=True)

# Initialiser le tracker avec la première image et la zone sélectionnée
ok = tracker.init(frame, bbox)

while True:
    # Lire une nouvelle image de la webcam
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la lecture de l'image.")
        break

    # Mettre à jour le suivi d'objet
    ok, bbox = tracker.update(frame)

    # Si le suivi est réussi, dessiner le rectangle autour de l'objet
    if ok:
        # bbox retourne (x, y, w, h) : coordonnées et dimensions du rectangle
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
    else:
        # Si l'objet est perdu
        cv2.putText(frame, "Objet perdu", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Afficher l'image avec le suivi
    cv2.imshow("Suivi d'objet en temps réel", frame)

    # Appuyer sur 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
