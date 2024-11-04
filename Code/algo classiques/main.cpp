#include "Image.hpp" // fichier de définition de la classe Image
#include <iostream>

#include "Filter.hpp"
#include "GaussianFilter.hpp"

int main()
{
    Image img("../../Ressources/In/Tumblr_l_2194398860526597.jpg");

    if (img.isLoaded())
    {
        img.printInfo(); // Affiche les informations de l'image

        Filter *filter = new GaussianFilter(1.0f); // Crée un filtre gaussien

        Image out = filter->apply(img); // Applique le filtre sur l'image

        out.savePNG("../../Ressources/Out/GaussianFilter.png"); // Sauvegarde l'image filtrée
    }
    else
    {
        std::cerr << "Échec du chargement de l'image." << std::endl;
    }
    return 0;
}
