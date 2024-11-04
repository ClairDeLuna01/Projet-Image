#include <iostream>
#include "include/Image.hpp"  // fichier de définition de la classe Image

int main() {
    Image img("chemin/vers/image.png");
    
    if (img.isLoaded()) {
        img.printInfo();              // Affiche les informations de l'image
        img.savePNG("output.png");    // Sauvegarde en PNG
    } else {
        std::cerr << "Échec du chargement de l'image." << std::endl;
    }
    return 0;
}
