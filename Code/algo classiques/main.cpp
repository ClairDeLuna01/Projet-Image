#include "extern/args.hxx"
#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "BilateralFilter.hpp"
#include "Filter.hpp"
#include "GaussianFilter.hpp"
#include "Image.hpp"
#include "Noiser.hpp"
#include "utils.hpp"

int main(int argc, char *argv[])
{
    // Définition des arguments
    args::ArgumentParser parser("Filtre d'image avec méthodes classiques");

    // Ajout des arguments
    args::Positional<std::string> filepathParam(parser, "filepath", "Chemin de l'image à traiter");
    args::MapPositional<std::string, FilterType> filterNameParam(
        parser, "filter", "Nom du filtre à appliquer (bilateral, gaussian)",
        {{"bilateral", FilterType::BILATERAL}, {"gaussian", FilterType::GAUSSIAN}}, FilterType::INVALID);

    args::ValueFlag<float> sigmaParam(parser, "sigma1", "Valeur de sigma pour le filtre", {'s', "sigma1", "sigma"},
                                      1.0f);
    args::ValueFlag<float> sigma2Param(parser, "sigma2", "Valeur de sigma2 pour le filtre bilatéral", {'t', "sigma2"},
                                       1.0f);
    args::ValueFlag<int> kernelSizeParam(parser, "kernelSize", "Taille du noyau pour le filtre bilatéral",
                                         {'k', "kernelSize"}, 3);

    args::HelpFlag help(parser, "help", "Affiche ce message d'aide", {'h', "help"});
    args::CompletionFlag completion(parser, {"complete"});

    // Parse les arguments
    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Completion)
    {
        std::cout << parser;
        return 0;
    }
    catch (args::Help)
    {
        std::cout << parser;
        return 0;
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    // Vérifie que le chemin de l'image est bien spécifié
    if (filepathParam.Get().empty())
    {
        std::cerr << "Erreur : Le chemin de l'image est obligatoire." << std::endl;
        return 1;
    }
    std::string filepath = filepathParam.Get();

    // Vérifie que le filtre est bien spécifié
    if (filterNameParam.Get() == FilterType::INVALID)
    {
        std::cerr << "Erreur : Le nom du filtre est obligatoire." << std::endl;
        return 1;
    }

    FilterType filterType = filterNameParam.Get();

    // Récupère les paramètres spécifiques à chaque filtre
    float sigma = sigmaParam.Get();
    float sigma2 = sigma2Param.Get();
    int kernelSize = kernelSizeParam.Get();

    // Charge l'image originale
    Image original(filepath);
    if (!original.isLoaded())
    {
        std::cerr << "Erreur : Échec du chargement de l'image originale." << std::endl;
        return 1;
    }
    std::string originalBaseName = utils::basenameWithoutExt(filepath);

    // Création du filtre selon le type entre en paramètre
    Filter *filter;
    std::string filterName = utils::filterTypeToString(filterType);

    switch (filterType)
    {
    case FilterType::BILATERAL:
        filter = new BilateralFilter(kernelSize, sigma, sigma2);
        break;
    case FilterType::GAUSSIAN:
        filter = new GaussianFilter(sigma);
        break;
    default:
        std::cerr << "Erreur : Filtre non reconnu." << std::endl;
        return 1;
    }

    // Map associant chaque type de bruit à sa fonction d'application
    std::map<NoiseType, std::function<void(Image &)>> noiseFunctions = {
        {NoiseType::GAUSSIAN, [](Image &img) { Noiser::applyGaussianNoise(img, 25.0); }},
        {NoiseType::POISSON, [](Image &img) { Noiser::applyPoissonNoise(img); }},
        {NoiseType::SALT_PEPPER, [](Image &img) { Noiser::applySaltAndPepperNoise(img, 0.05); }},
        {NoiseType::SPECKLE, [](Image &img) { Noiser::applySpeckleNoise(img, 0.1); }}};

    for (const auto &[noiseType, applyNoise] : noiseFunctions)
    {
        Image noisyImage = original; // Copie de l'image d'origine
        applyNoise(noisyImage);      // Applique le bruit correspondant

        // Sauvegarde de l'image bruitée avec le nom du bruit
        std::string noiseName = utils::noiseTypeToString(noiseType);
        noisyImage.savePNG("../../Ressources/Out/" + originalBaseName + "_" + noiseName + "_noise.png");

        if (noisyImage.isLoaded())
        {
            Image out = filter->apply(noisyImage);               // Applique le filtre
            float psnrNoisy = utils::PSNR(noisyImage, original); // PSNR de l'image bruitée
            float psnrFiltered = utils::PSNR(out, original);     // PSNR de l'image filtrée
            float psnrDiff = psnrFiltered - psnrNoisy;

            float ssimNoisy = utils::SSIM(noisyImage, original); // SSIM de l'image bruitée
            float ssimFiltered = utils::SSIM(out, original);     // SSIM de l'image filtrée
            float ssimDiff = ssimFiltered - ssimNoisy;

            std::cout << "================================\n";
            std::cout << "Image bruitée par " << noiseName << ":\n";
            std::cout << "PSNR (bruitée): " << psnrNoisy << " dB\n";
            std::cout << "PSNR (filtrée): " << psnrFiltered << " dB\n";
            std::cout << "Différence de PSNR : " << psnrDiff << " dB\n";
            std::cout << "SSIM (bruitée): " << ssimNoisy << "\n";
            std::cout << "SSIM (filtrée): " << ssimFiltered << "\n";
            std::cout << "Différence de SSIM : " << ssimDiff << "\n";

            out.savePNG("../../Ressources/Out/" + originalBaseName + "_" + noiseName + "_denoisedBy" + filterName +
                        ".png");
        }
    }

    delete filter; // Libération de la mémoire du filtre

    return 0; // Succès
}
