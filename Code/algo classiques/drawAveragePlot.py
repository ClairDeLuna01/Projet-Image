import os
import matplotlib.pyplot as plt
import numpy as np

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

# Fonction pour calculer la moyenne des métriques par méthode de filtrage
def compute_average_metric_per_filter(scene_names, base_dir, filters, col_x, col_y):
    """Calcule la moyenne des métriques pour chaque méthode de filtrage."""
    averages_per_filter = {}

    for filter_name in filters:
        aggregated_data = {}  # Dictionnaire pour stocker les données par valeur de x

        for scene_name in scene_names:
            filepath = os.path.join(base_dir, scene_name, f"{scene_name}_denoisedBy{filter_name}.dat")
            x_values, y_values = read_data(filepath, col_x, col_y)

            for x, y in zip(x_values, y_values):
                if x not in aggregated_data:
                    aggregated_data[x] = []
                aggregated_data[x].append(y)

        # Calculer la moyenne pour chaque valeur de x
        averaged_x = sorted(aggregated_data.keys())
        averaged_y = [np.mean(aggregated_data[x]) for x in averaged_x]

        averages_per_filter[filter_name] = (averaged_x, averaged_y)

    return averages_per_filter

# Fonction pour tracer les courbes moyennes par méthode de filtrage
def plot_average_per_filter(averages_per_filter, metric_name, output_file):
    """Trace les courbes moyennes par méthode de filtrage pour une métrique donnée."""
    plt.figure(figsize=(10, 6))
    plt.title(f"Average {metric_name} vs SPP per Filter")
    plt.xlabel("Samples per Pixel (SPP)")
    plt.ylabel(f"Average {metric_name} (%)")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Marqueurs sur l'axe x de 64 en 64
    plt.xticks(range(64, 1025, 64))

    # Tracer une courbe pour chaque méthode de filtrage
    for filter_name, (x_values, y_values) in averages_per_filter.items():
        plt.plot(x_values, y_values, marker="o", label=filter_name)

    plt.legend()
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()
    print(f"Graphique sauvegardé : {output_file}")

# Calcul et tracé des moyennes pour le PSNR et le SSIM
psnr_averages = compute_average_metric_per_filter(SCENE_NAMES, BASE_DIR, FILTERS, col_x=0, col_y=3)
ssim_averages = compute_average_metric_per_filter(SCENE_NAMES, BASE_DIR, FILTERS, col_x=0, col_y=6)

# Sauvegarder les graphiques
plot_average_per_filter(psnr_averages, "PSNR Gain", os.path.join(BASE_DIR, "average_psnr_per_filter.png"))
plot_average_per_filter(ssim_averages, "SSIM", os.path.join(BASE_DIR,"average_ssim_per_filter.png"))

print("Graphiques des moyennes par méthode de filtrage générés.")
