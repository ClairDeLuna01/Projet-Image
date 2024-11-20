import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Masquer les messages d'avertissement
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tkinter import Tk, filedialog
from sys import argv

# Fonction pour ouvrir une boîte de dialogue de sélection de fichier
def select_image():
    root = Tk()
    root.withdraw()  # Masquer la fenêtre principale de tkinter
    file_path = filedialog.askopenfilename(
        title="Sélectionner une image",
        initialdir="./../../Ressources",
        filetypes=[("Fichiers image", "*.jpg *.jpeg *.png *.ppm")]
    )
    root.destroy()  # Fermer la fenêtre principale après sélection
    return file_path

# Charger et préparer le modèle NIMA
def load_nima_model():
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    # Charger les poids pré-entraînés si disponibles
    # model.load_weights('path/to/nima_weights.h5')
    return model

# Prédire le score NIMA pour une image donnée
def evaluate_image_quality(model, image_path):
    # Charger et prétraiter l'image
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Obtenir la distribution des scores
    predicted_scores = model.predict(image)[0]
    
    # Calculer le score moyen NIMA
    nima_score = sum((i + 1) * predicted_scores[i] for i in range(10))
    return nima_score

# Sélectionner l'image et calculer le score NIMA
# image_path = select_image()


image_path = argv[1] if len(argv) > 1 else select_image()
if image_path:
    model = load_nima_model()
    score = evaluate_image_quality(model, image_path)
    print(f"NIMA: {score}")
else:
    print("Aucune image sélectionnée.")
