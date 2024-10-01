# Partie 1 : Exercice 2

import cv2
import numpy as np

# Étape 1 : Charger l'image
image = cv2.imread('mount-fuji.png')

# Étape 2 : Définir la ROI (Region of Interest) manuellement
x, y, w, h = 100, 100, 200, 200  # Exemple de coordonnées pour la ROI
roi = image[y:y+h, x:x+w]

# Étape 3 : Appliquer un masque à la ROI (ex. augmenter la luminosité)
mask = np.ones(roi.shape, dtype="uint8") * 50
roi_bright = cv2.add(roi, mask)  # Augmente la luminosité

# Remplacer la ROI par la version modifiée dans l'image d'origine
image[y:y+h, x:x+w] = roi_bright

# Étape 4 : Appliquer un flou gaussien sur l'image entière, sauf la ROI
image_flou = cv2.GaussianBlur(image, (21, 21), 0)

# Remettre la ROI non floutée à sa place dans l'image floutée
image_flou[y:y+h, x:x+w] = roi_bright

# Étape 5 : Afficher et sauvegarder l'image
cv2.imshow('Image modifiée', image_flou)
cv2.imwrite('image_modifiee.jpg', image_flou)

cv2.waitKey(0)
cv2.destroyAllWindows()
