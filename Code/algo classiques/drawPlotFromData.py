import os
import matplotlib.pyplot as plt

# Liste des scènes
SCENE_NAMES = [
    "bathroom", "bathroom2", "bedroom", "classroom", "cornell-box",
    "dining-room", "glass-of-water", "kitchen", "living-room",
    "living-room-2", "living-room-3", "staircase", "staircase2",
    "veach-ajar", "veach-bidir", "water-caustic", "volumetric-caustic"
]

# Chemin de base des fichiers
BASE_DIR = "../../Ressources/Out"

# Liste des filtres (types de débruiteurs)
FILTERS = [
    "bilateral", "ffdnet_pretrained", "gaussian", "median"
]

# Fonction pour lire un fichier de données
def read_data(filepath, col_x, col_y):
    """Lit les données d'un fichier .dat et renvoie deux listes : col_x et col_y."""
    x = []
    y = []
    try:
        with open(filepath, "r") as file:
            for line in file:
                columns = line.strip().split()
                if len(columns) <= max(col_x, col_y):  # Vérifiez que les colonnes existent
                    continue
                x.append(float(columns[col_x]))  # Colonne pour l'axe X
                y.append(float(columns[col_y]))  # Colonne pour l'axe Y
    except FileNotFoundError:
        print(f"Fichier non trouvé : {filepath}")
    except Exception as e:
        print(f"Erreur lors de la lecture de {filepath}: {e}")
    return x, y

# Fonction pour tracer les courbes PSNR
def plot_metric(scene_name, base_dir, filters, metric_name, col_y, output_suffix):
    """Trace les courbes d'une métrique donnée pour une scène."""
    plt.figure(figsize=(10, 6))
    plt.title(f"{metric_name} vs SPP for {scene_name}")
    plt.xlabel("Samples per Pixel (SPP)")
    plt.ylabel(f"{metric_name} (%)")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Marqueurs sur l'axe x de 64 en 64
    plt.xticks(range(64, 1025, 64))

    for filter_name in filters:
        filepath = os.path.join(base_dir, scene_name, f"{scene_name}_denoisedBy{filter_name}.dat")
        spp, metric_values = read_data(filepath, col_x=0, col_y=col_y)
        if spp and metric_values:  # Vérifiez si des données ont été chargées
            plt.plot(spp, metric_values, label=filter_name)

    plt.legend()
    plt.tight_layout()

    # Sauvegarder l'image dans le dossier de la scène
    output_file = os.path.join(base_dir, scene_name, f"{scene_name}_{output_suffix}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Graphique sauvegardé : {output_file}")

# Boucle sur chaque scène
for scene in SCENE_NAMES:
    # Tracer les courbes pour le PSNR (colonne 0 et 4)
    plot_metric(scene, BASE_DIR, FILTERS, "PSNR Gain", col_y=3, output_suffix="psnr_plot")

    # Tracer les courbes pour le SSIM (colonne 0 et 6)
    plot_metric(scene, BASE_DIR, FILTERS, "SSIM Gain", col_y=6, output_suffix="ssim_plot")

print("Tous les graphiques PSNR et SSIM ont été générés et sauvegardés.")
