#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "Image.hpp"
#include "BilinearFilter.hpp"
#include "Filter.hpp"
#include "GaussianFilter.hpp"
#include "utils.hpp"
#include "Noiser.hpp"

int main(int argc, char *argv[])
{
    // Vérifier si un chemin de fichier est passé en argument
    if (argc < 4)
    {
        std::cerr << "Utilisation : " << argv[0] << " <filepath> <filter> <sigma>" << std::endl;
        return 1;
    }

    // Récupérer le chemin de l'image depuis les arguments de la ligne de commande
    std::string filepath = argv[1];
    std::string filterName = argv[2];
    std::transform(filterName.begin(), filterName.end(), filterName.begin(), ::tolower);
    std::string sigmaStr = argv[3];
    float sigma = std::stof(sigmaStr);

    // Compute le type de filtre selon le nom du filtre passé en paramètre
    FilterType filterType;
    if (filterName == "bilinear")
    {
        filterType = FilterType::BILINEAR;
        std::cout << "Filtre bilinéaire" << std::endl;
    }
    else if (filterName == "gaussian")
    {
        filterType = FilterType::GAUSSIAN;
        std::cout << "Filtre gaussien" << std::endl;
    }
    else
    {
        std::cerr << "Erreur : Filtre non reconnu." << std::endl;
        return 1;
    }

    // Charge l'image originale
    Image original(filepath);
    if (!original.isLoaded()) {
        std::cerr << "Erreur : Échec du chargement de l'image originale." << std::endl;
        return 1;
    }
    std::string originalBaseName = utils::basenameWithoutExt(filepath);

    // Création du filtre selon le type entre en paramètre
    Filter *filter;

    switch (filterType)
    {
    case FilterType::BILINEAR:
        filter = new BilinearFilter();
        break;
    case FilterType::GAUSSIAN:
        filter = new GaussianFilter(sigma);
        break;
    }

    // Map associant chaque type de bruit à sa fonction d'application
    std::map<NoiseType, std::function<void(Image&)>> noiseFunctions = {
        { NoiseType::GAUSSIAN,    [](Image& img) { Noiser::applyGaussianNoise(img, 25.0); } },
        { NoiseType::POISSON,     [](Image& img) { Noiser::applyPoissonNoise(img); } },
        { NoiseType::SALT_PEPPER, [](Image& img) { Noiser::applySaltAndPepperNoise(img, 0.05); } },
        { NoiseType::SPECKLE,     [](Image& img) { Noiser::applySpeckleNoise(img, 0.1); } }
    };

    for (const auto& [noiseType, applyNoise] : noiseFunctions) {
        Image noisyImage = original;  // Copie de l'image d'origine
        applyNoise(noisyImage);       // Applique le bruit correspondant

        // Sauvegarde de l'image bruitée avec le nom du bruit
        std::string noiseName = utils::noiseTypeToString(noiseType);
        noisyImage.savePNG("../../Ressources/Out/" + originalBaseName + "_" + noiseName + "_noise.png");

        if (noisyImage.isLoaded()) {
            Image out = filter->apply(noisyImage); // Applique le filtre
            float psnrNoisy = utils::PSNR(noisyImage, original);       // PSNR de l'image bruitée
            float psnrFiltered = utils::PSNR(out, original);    // PSNR de l'image filtrée
            float psnrDiff = psnrFiltered - psnrNoisy;

            std::cout << "Image bruitée par " << noiseName << ":\n";
            std::cout << "PSNR (bruitée): " << psnrNoisy << " dB\n";
            std::cout << "PSNR (filtrée): " << psnrFiltered << " dB\n";
            std::cout << "Différence de PSNR : " << psnrDiff << " dB\n\n";

            out.savePNG("../../Ressources/Out/" + originalBaseName + "_" + noiseName + "_denoisedBy" + filterName + ".png");
        }
    }

    delete filter; // Libération de la mémoire du filtre

    return 0; // Succès
}
