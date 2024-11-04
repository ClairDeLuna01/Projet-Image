import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_poisson_noise(image):
    noisy_image = np.random.poisson(image / 255.0 * 30) / 30 * 255
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, amount=0.05, salt_vs_pepper=0.5):
    noisy_image = image.copy()
    num_salt = int(amount * image.size * salt_vs_pepper)
    num_pepper = int(amount * image.size * (1 - salt_vs_pepper))

    # Ajouter le sel (pixels blancs)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1], :] = 255

    # Ajouter le poivre (pixels noirs)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1], :] = 0

    return noisy_image

def add_speckle_noise(image):
    noise = np.random.randn(*image.shape).astype(np.float32)
    noisy_image = image + image * noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_blue_noise(image):
    # Transformer l'image dans le domaine fréquentiel pour chaque canal
    noisy_image = np.zeros_like(image, dtype=np.float32)
    for channel in range(3):
        f_transform = fft2(image[:, :, channel])
        f_shifted = fftshift(f_transform)

        # Créer un masque pour atténuer les hautes fréquences
        rows, cols = image.shape[:2]
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.float32)
        for u in range(rows):
            for v in range(cols):
                dist = np.sqrt((u - crow)**2 + (v - ccol)**2)
                mask[u, v] = 1.0 / (1.0 + dist)

        # Appliquer le masque et transformer à nouveau dans le domaine spatial
        f_shifted = f_shifted * mask
        f_ishifted = ifftshift(f_shifted)
        noisy_channel = ifft2(f_ishifted).real
        noisy_image[:, :, channel] = noisy_channel

    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def save_and_show_images():
    # Demander le chemin de l'image en entrée
    image_path = input("Entrez le chemin de l'image : ")
    
    # Charger l'image en couleur
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Erreur : L'image n'a pas pu être chargée. Vérifiez le chemin.")
        return

    # Appliquer différents bruits
    gaussian_image = add_gaussian_noise(original_image)
    poisson_image = add_poisson_noise(original_image)
    salt_pepper_image = add_salt_and_pepper_noise(original_image)
    speckle_image = add_speckle_noise(original_image)
    blue_image = add_blue_noise(original_image)

    # Sauvegarder et afficher les images bruitées
    noises = {'Original': original_image, 'Gaussian': gaussian_image, 'Poisson': poisson_image,
              'Salt_Pepper': salt_pepper_image, 'Speckle': speckle_image, 'Blue': blue_image}
    
    # Créer un dossier "output" pour enregistrer les images
    import os
    if not os.path.exists("output"):
        os.makedirs("output")

    for noise_type, img in noises.items():
        output_path = f"output/{noise_type}_noise.png"
        cv2.imwrite(output_path, img)
        print(f"Image avec {noise_type} bruit sauvegardée sous : {output_path}")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{noise_type} Noise")
        plt.axis('off')
        plt.show()

# Exécuter la fonction principale
save_and_show_images()
