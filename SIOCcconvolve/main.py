# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
from skimage import io
from scipy.ndimage import convolve
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Wczytanie pierwszego obrazu i wyświetlenie jego rozmiaru
image = io.imread(r"C:/Users/Szymon Nowicki/Desktop/CFA/Bayer/circle.jpg")
image.shape

# Wyświetlenie pierwszego obrazu
io.imshow(image)

# Konwersja obrazu na skalę szarości
gray_image = rgb2gray(image)

# Definiowanie filtru Laplace'a
laplace_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# Definiowanie filtrów Sobela dla krawędzi poziomych i pionowych
sobel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Stosowanie filtru Laplace'a do każdego kanału koloru obrazu
filtered_image = np.dstack([
    convolve(image[:, :, channel], laplace_filter, mode="constant", cval=0.0)
    for channel in range(3)
])

# Stosowanie filtrów Sobela
filtered_sobel_horizontal = convolve(gray_image, sobel_horizontal, mode='constant', cval=0.0)
filtered_sobel_vertical = convolve(gray_image, sobel_vertical, mode='constant', cval=0.0)

# Obliczanie całkowitej wartości gradientu
sobel_gradient = np.sqrt(filtered_sobel_horizontal**2 + filtered_sobel_vertical**2)

# Wyświetlenie obrazów dla porównania
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Oryginalny obraz')

plt.subplot(1, 3, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtr Laplace\'a')

plt.subplot(1, 3, 3)
plt.imshow(sobel_gradient, cmap='gray')
plt.title('Filtr Sobela')

plt.show()

# Wczytanie drugiego obrazu i wyświetlenie jego rozmiaru
image = io.imread(r"C:/Users/Szymon Nowicki/Desktop/CFA/Bayer/panda.jpg")
image.shape

# Wyświetlenie drugiego obrazu
io.imshow(image)

# Definiowanie średniego filtru (mean filter)
mean_filter = np.array([[1, 2, 1], [1, 4, 1], [1, 2, 1]]) / 16

# Alternatywna definicja średniego filtru
mean_filter = np.ones([9, 9]) / (9 ** 2)
mean_filter.shape

# Stosowanie średniego filtru do każdego kanału koloru obrazu
filtered_image = np.dstack([
    convolve(image[:, :, channel], mean_filter, mode="constant", cval=0.0)
    for channel in range(3)
])

# Wyświetlenie rozmiaru przefiltrowanego obrazu
filtered_image.shape

# Wyświetlenie przefiltrowanego obrazu
io.imshow(filtered_image)

# Definiowanie filtru wyostrzającego
mean_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
mean_filter = mean_filter / mean_filter.sum()

# Konwersja obrazu na skalę szarości
image = rgb2gray(image)
image.shape

# Wyświetlenie obrazu w skali szarości
io.imshow(image)

# Ponowne wyświetlenie filtru wyostrzającego
mean_filter

# Stosowanie filtru wyostrzającego do obrazu w skali szarości
filtered_image = convolve(image, mean_filter, mode="constant", cval=0.0)
filtered_image.shape

# Normalizacja przefiltrowanego obrazu
filtered_image = filtered_image - filtered_image.min()
filtered_image = filtered_image / filtered_image.max()

# Wyświetlenie przefiltrowanego i znormalizowanego obrazu w skali szarości
io.imshow(filtered_image, cmap="Greys")

# Wyświetlenie obrazów za pomocą matplotlib
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Oryginalny obraz')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(filtered_image)
plt.title('Obraz znormalizowany i przefiltrowany')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap="gray")
plt.title('Obraz Obraz znormalizowany i przefiltrowany w skali szarości')
plt.axis('off')

plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
