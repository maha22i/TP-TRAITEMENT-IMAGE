# Partie 1 : Exercice 1

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image de haute résolution
image = cv2.imread('mount-fuji.png', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Erreur : Impossible de charger l'image. Vérifiez le chemin du fichier.")
    exit()

# Appliquer le filtre Sobel pour détecter les contours
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Contours verticaux
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Contours horizontaux
sobel = cv2.magnitude(sobelx, sobely)

# Afficher et sauvegarder les résultats du filtre Sobel
cv2.imshow('Sobel Contours', sobel)
cv2.imwrite('sobel_contours.png', sobel)

# Transformation de Fourier
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Manipulation du spectre (par exemple, suppression du centre)
rows, cols = image.shape
crow, ccol = rows // 2 , cols // 2
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

# Transformation inverse de Fourier
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Afficher et sauvegarder les résultats de la transformation de Fourier
cv2.imshow('Magnitude Spectrum', magnitude_spectrum)
cv2.imwrite('magnitude_spectrum.png', magnitude_spectrum)

cv2.imshow('Inverse Fourier Transform', img_back)
cv2.imwrite('inverse_fourier.png', img_back)

# Segmentation par seuillage adaptatif
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Afficher et sauvegarder les résultats du seuillage adaptatif
cv2.imshow('Seuillage Adaptatif', thresh)
cv2.imwrite('seuillage_adaptatif.png', thresh)

# Attendre une touche pour fermer toutes les fenêtres
cv2.waitKey(0)
cv2.destroyAllWindows()
