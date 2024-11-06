#include "Image.hpp" // fichier de définition de la classe Image
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "BilinearFilter.hpp"
#include "Filter.hpp"
#include "GaussianFilter.hpp"
#include "utils.hpp"

enum class FilterType
{
    BILINEAR,
    GAUSSIAN
};

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
    std::string filter = argv[2];
    std::transform(filter.begin(), filter.end(), filter.begin(), ::tolower);
    std::string sigmaStr = argv[3];
    float sigma = std::stof(sigmaStr);

    // Récupérer tout les images dans le dossier
    std::vector<std::string> files = utils::getFilesInDirectory("../../Ressources/In/" + filepath);

    FilterType filterType;
    if (filter == "bilinear")
    {
        filterType = FilterType::BILINEAR;
        std::cout << "Filtre bilinéaire" << std::endl;
    }
    else if (filter == "gaussian")
    {
        filterType = FilterType::GAUSSIAN;
        std::cout << "Filtre gaussien" << std::endl;
    }
    else
    {
        std::cerr << "Erreur : Filtre non reconnu." << std::endl;
        return 1;
    }

    std::string originalFile;
    for (const std::string &file : files)
    {
        if (file.find("Original") != std::string::npos)
        {
            originalFile = file;
            break;
        }
    }

    if (originalFile.empty())
    {
        std::cerr << "Erreur : Aucune image originale trouvée." << std::endl;
        return 1;
    }

    Image original("../../Ressources/In/" + filepath + "/" + originalFile);

    for (const std::string &file : files)
    {
        // if (file.find("Original") != std::string::npos)
        // {
        //     continue;
        // }

        std::string path = "../../Ressources/In/" + filepath + "/" + file;

        std::string outFolder;
        switch (filterType)
        {
        case FilterType::BILINEAR:
            outFolder = filepath + "_bilinear";
            break;
        case FilterType::GAUSSIAN:
            outFolder = filepath + "_gaussian";
            break;
        }
        std::string path_out = "../../Ressources/Out/" + outFolder + "/" + utils::basename(file) + "_out.png";

        // Créer une instance de l'image et la charger
        Image img(path);

        if (img.isLoaded())
        {
            // img.printInfo(); // Affiche les informations de l'image

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

            Image out = filter->apply(img); // Applique le filtre sur l'image

            float psnr = utils::PSNR(img, original); // Calcule le PSNR entre l'image filtrée et l'image originale
            float psnrFilter = utils::PSNR(out, original); // Calcule le PSNR entre l'image filtrée et l'image originale
            float psnrDiff = psnrFilter - psnr;

            std::cout << "Image : " << file << std::endl;

            std::cout << "PSNR (bruitée): " << psnr << " dB" << std::endl;
            std::cout << "PSNR (filtre) : " << psnrFilter << " dB" << std::endl;
            std::cout << "Différence de PSNR : " << psnrDiff << " dB" << std::endl << std::endl;

            out.savePNG(path_out); // Sauvegarde l'image filtrée
        }
        else
        {
            std::cerr << "Erreur : Échec du chargement de l'image depuis le chemin spécifié." << std::endl;
            return 1; // Retourner une erreur si l'image n'a pas pu être chargée
        }
    }

    return 0; // Succès
}
