#include "Image.hpp" // fichier de définition de la classe Image
#include <iostream>

#include "Filter.hpp"
#include "GaussianFilter.hpp"
int main(int argc, char *argv[])
{
    // Vérifier si un chemin de fichier est passé en argument
    if (argc < 2)
    {
        std::cerr << "Utilisation : " << argv[0] << " <filepath>" << std::endl;
        return 1;
    }

    // Récupérer le chemin de l'image depuis les arguments de la ligne de commande
    std::string filepath = argv[1];

    // Créer une instance de l'image et la charger
    Image img(filepath);

    if (img.isLoaded())
    {
        img.printInfo(); // Affiche les informations de l'image

        Filter *filter = new GaussianFilter(1.0f); // Crée un filtre gaussien

        Image out = filter->apply(img); // Applique le filtre sur l'image

        out.savePNG("../../Ressources/Out/GaussianFilter.png"); // Sauvegarde l'image filtrée
    }
    else
    {
        std::cerr << "Erreur : Échec du chargement de l'image depuis le chemin spécifié." << std::endl;
        return 1; // Retourner une erreur si l'image n'a pas pu être chargée
    }

    return 0; // Succès
}
