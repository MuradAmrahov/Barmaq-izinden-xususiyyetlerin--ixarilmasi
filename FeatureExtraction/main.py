from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Şəkli yüklə
image_path = "C:/Users/user/Desktop/DB4_B/102_8.tif" #Barmaq izi şəklinin yolu
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# ORB xüsusiyyət çıxarıcı obyektini yarat
orb = cv2.ORB_create()

# Xüsusiyyətləri çıxar
keypoints = orb.detect(image, None)

# Xüsusiyyətləri şəkilə çək
output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

# Şəkili göstər
plt.figure(figsize=(10, 10))
plt.imshow(output_image, cmap='gray')
plt.axis('off')
plt.show()

