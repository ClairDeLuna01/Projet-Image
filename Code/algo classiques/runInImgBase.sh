#!/bin/bash

# Vérification des arguments
if [[ $# -lt 1 ]]; then
    echo "Usage: ./process_scenes.sh <denoiser_type>"
    echo "Exemple : ./process_scenes.sh ffdnet_pretrained"
    exit 1
fi

# Récupérer le type de débruiteur (comme ffdnet_pretrained, bilateral, gaussian, etc.)
DENOISER_TYPE=$1

# Définir les noms des scènes
SCENE_NAMES=(
    "bathroom" "bathroom2" "bedroom" "classroom" "cornell-box"
    "dining-room" "glass-of-water" "kitchen" "living-room"
    "living-room-2" "living-room-3" "staircase" "staircase2"
    "veach-ajar" "veach-bidir" "water-caustic" "volumetric-caustic"
)

# Liste des samples (par exemple, les différentes valeurs d'échantillons)
SAMPLES=(64 128 192 256 320 384 448 512 576 640 704 768 832 896 960 1024)

# Répertoires de base
IMG_BASE_DIR="../../Ressources/ImgBase"
OUT_BASE_DIR="../../Ressources/Out"

# Boucle sur chaque scène
for scene_name in "${SCENE_NAMES[@]}"; do
    # Boucle sur chaque sample
    for sample in "${SAMPLES[@]}"; do
        # Chemins source et destination
        input_file="$IMG_BASE_DIR/$scene_name/scene_spp_$sample/$scene_name.png"
        output_file="$OUT_BASE_DIR/$scene_name/${scene_name}_${sample}_denoisedBy${DENOISER_TYPE}.png"
        original_file="$IMG_BASE_DIR/$scene_name/${scene_name}_original.png"

        # Vérifier si le fichier d'entrée existe
        if [[ ! -f "$input_file" ]]; then
            echo "Fichier introuvable : $input_file"
            continue
        fi

        # Créer les dossiers nécessaires si inexistants
        mkdir -p "$(dirname "$output_file")"
        mkdir -p "$(dirname "$original_file")"

        # Lancer la commande avec le débruiteur dynamique
        echo "Lancement de : ./main $input_file $DENOISER_TYPE -n -o $original_file -d $output_file"
        ./main "$input_file" "$DENOISER_TYPE" -n -o "$original_file" -d "$output_file"
    done
done

echo "Traitement terminé pour toutes les scènes et tous les samples."
