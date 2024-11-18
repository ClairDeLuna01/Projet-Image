import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

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

def choose_file_and_folder():
    # Initialiser Tkinter et masquer la fenêtre principale
    root = Tk()
    root.withdraw()

    # Demander à l'utilisateur de sélectionner un fichier d'image
    image_path = filedialog.askopenfilename(title="Sélectionnez le fichier d'image", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
    if not image_path:
        print("Aucun fichier sélectionné.")
        return None, None

    # Demander à l'utilisateur de sélectionner un dossier de sortie
    output_folder = filedialog.askdirectory(title="Sélectionnez le dossier de sortie")
    if not output_folder:
        print("Aucun dossier de sortie sélectionné.")
        return None, None

    return image_path, output_folder

def save_and_show_images():
    # Choisir le fichier d'image et le dossier de sortie
    image_path, output_folder = choose_file_and_folder()
    if not image_path or not output_folder:
        return

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

    # Sauvegarder et afficher les images bruitées
    noises = {'Original': original_image, 'Gaussian': gaussian_image, 'Poisson': poisson_image,
              'Salt_Pepper': salt_pepper_image, 'Speckle': speckle_image}

    # Créer un dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for noise_type, img in noises.items():
        output_path = os.path.join(output_folder, f"{noise_type}_noise.png")
        cv2.imwrite(output_path, img)
        print(f"Image avec {noise_type} bruit sauvegardée sous : {output_path}")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{noise_type} Noise")
        plt.axis('off')
        plt.show()

# Exécuter la fonction principale
save_and_show_images()
