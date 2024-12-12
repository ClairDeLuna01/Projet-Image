#include "extern/args.hxx"
#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "BilateralFilter.hpp"
#include "FFDNet.hpp"
#include "FFDNetPretrained.hpp"
#include "Filter.hpp"
#include "GaussianFilter.hpp"
#include "Image.hpp"
#include "MedianFilter.hpp"
#include "Noiser.hpp"
#include "NonlocalMeansFilter.hpp"
#include "utils.hpp"

int main(int argc, char *argv[])
{
    // Définition des arguments
    args::ArgumentParser parser("Filtre d'image avec méthodes classiques");

    // Ajout des arguments
    args::Positional<std::string> filepathParam(parser, "filepath", "Chemin de l'image à traiter");
    args::MapPositional<std::string, FilterType> filterNameParam(
        parser, "filter",
        "Nom du filtre à appliquer (bilateral, gaussian, nonlocal_means, median, ffdnet_pretrained, ffdnet)",
        {{"bilateral", FilterType::BILATERAL},
         {"gaussian", FilterType::GAUSSIAN},
         {"nonlocal_means", FilterType::NONLOCAL_MEANS},
         {"median", FilterType::MEDIAN},
         {"ffdnet_pretrained", FilterType::FFDNET_PRETRAINED},
         {"ffdnet", FilterType::FFDNET}},
        FilterType::INVALID);
    args::Flag noAddNoiseParam(parser, "noAddNoise", "Ne pas ajouter de bruit à l'image", {'n', "noAddNoise"});
    args::Flag compactPrint(parser, "compactPrint", "Affichage compact terminal", {'c', "compactPrint"});
    args::ValueFlag<std::string> originalPathParam(parser, "originalPath", "Chemin de l'image originale",
                                                   {'o', "originalPath"}, "");
    args::ValueFlag<std::string> outputPathParam(parser, "outputPath", "Chemin de l'image de sortie",
                                                 {'d', "outputPath"}, "");
    args::ValueFlag<float> sigmaParam(parser, "sigma1", "Valeur de sigma pour le filtre", {'s', "sigma1", "sigma"},
                                      -1.0f);
    args::ValueFlag<float> sigma2Param(parser, "sigma2", "Valeur de sigma2 pour le filtre bilatéral", {'t', "sigma2"},
                                       -1.0f);
    args::ValueFlag<float> sigmaGaussianParam(parser, "sigmaGaussian", "Valeur de sigmaGaussian", {"sigmaGaussian"},
                                              25.0f);
    args::ValueFlag<float> sigmaSpeckleParam(parser, "sigmaSpeckle", "Valeur de sigmaSpeckle", {"sigmaSpeckle"}, 0.1f);
    args::ValueFlag<float> sigmaSaltAndPepperParam(parser, "sigmaSaltAndPepper", "Valeur de sigmaSaltAndPepper",
                                                   {"sigmaSaltAndPepper"}, 0.05f);
    args::ValueFlag<int> kernelSizeParam(parser, "kernelSize", "Taille du noyau pour le filtre bilatéral",
                                         {'k', "kernelSize"}, -1);
    args::ValueFlag<int> iterNbrParam(parser, "iterNbr", "Nombre d'itération de filtrage",
                                      {'i', "iterNbr", "iterationNbr"}, 1);
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
    float sigmaGaussian = sigmaGaussianParam.Get();
    float sigmaSpeckle = sigmaSpeckleParam.Get();
    float sigmaSaltAndPepper = sigmaSaltAndPepperParam.Get();
    int kernelSize = kernelSizeParam.Get();
    int iterNbr = iterNbrParam.Get();

    // Création du filtre selon le type entre en paramètre
    Filter *filter;
    std::string filterName = utils::filterTypeToString(filterType);

    switch (filterType)
    {
    case FilterType::BILATERAL:
        sigma = sigma == -1.0f ? 1.2 : sigma;
        sigma2 = sigma2 == -1.0f ? 120.0 : sigma2;
        kernelSize = kernelSize == -1.0f ? 7 : kernelSize;
        filter = new BilateralFilter(kernelSize, sigma, sigma2);
        break;
    case FilterType::GAUSSIAN:
        sigma = sigma == -1.0f ? 1.54 : sigma;
        filter = new GaussianFilter(sigma);
        break;
    case FilterType::NONLOCAL_MEANS:
        sigma = sigma == -1.0f ? 1.2 : sigma;
        kernelSize = kernelSize == -1.0f ? 7 : kernelSize;
        filter = new NonlocalMeansFilter(sigma, kernelSize);
        break;
    case FilterType::MEDIAN:
        kernelSize = kernelSize == -1.0f ? 3 : kernelSize;
        filter = new MedianFilter(kernelSize);
        break;
    case FilterType::FFDNET_PRETRAINED:
        filter = new FFDNetPretrained();
        break;
    case FilterType::FFDNET:
        filter = new FFDNet();
        break;
    default:
        std::cerr << "Erreur : Filtre non reconnu." << std::endl;
        return 1;
    }

    // Map associant chaque type de bruit à sa fonction d'application
    std::map<NoiseType, std::function<void(Image &)>> noiseFunctions = {
        {NoiseType::GAUSSIAN, [sigmaGaussian](Image &img) { Noiser::applyGaussianNoise(img, sigmaGaussian); }},
        {NoiseType::POISSON, [](Image &img) { Noiser::applyPoissonNoise(img); }},
        {NoiseType::SALT_PEPPER,
         [sigmaSaltAndPepper](Image &img) { Noiser::applySaltAndPepperNoise(img, sigmaSaltAndPepper); }},
        {NoiseType::SPECKLE, [sigmaSpeckle](Image &img) { Noiser::applySpeckleNoise(img, sigmaSpeckle); }}};

    if (!noAddNoiseParam.Get())
    {
        // if (compactPrint.Get())
        // {
        //     std::cout << "filterType psnrGain ssimGain\n";
        // }

        // Charge l'image originale
        Image original(filepath);
        if (!original.isLoaded())
        {
            std::cerr << "Erreur : Échec du chargement de l'image originale." << std::endl;
            return 1;
        }
        std::string originalBaseName = utils::basenameWithoutExt(filepath);
        for (const auto &[noiseType, applyNoise] : noiseFunctions)
        {
            Image noisyImage = original; // Copie de l'image d'origine
            applyNoise(noisyImage);      // Applique le bruit correspondant
            // Sauvegarde de l'image bruitée avec le nom du bruit
            std::string noiseName = utils::noiseTypeToString(noiseType);
            // noisyImage.savePNG("../../Ressources/Out/" + originalBaseName + "_" + noiseName + "_noise.png");
            float noiseSigma = (noiseType == NoiseType::GAUSSIAN)      ? sigmaGaussian
                               : (noiseType == NoiseType::SALT_PEPPER) ? sigmaSaltAndPepper
                                                                       : sigmaSpeckle;
            if ((noiseType != NoiseType::POISSON))
                noisyImage.savePNG("../../Ressources/Out/" + originalBaseName + "_" + noiseName + "_" +
                                   std::to_string(noiseSigma) + ".png");
            if (noisyImage.isLoaded())
            {
                Image out = noisyImage;
                for (int i = 0; i < iterNbr; i++)
                {
                    out = filter->apply(out); // Applique le filtre
                }
                float psnrNoisy = utils::PSNR(noisyImage, original); // PSNR de l'image bruitée
                float psnrFiltered = utils::PSNR(out, original);     // PSNR de l'image filtrée
                // float psnrDiff = psnrFiltered - psnrNoisy;
                float psnrGain = (psnrFiltered / psnrNoisy) - 1.0f;

                float ssimNoisy = utils::SSIM(noisyImage, original); // SSIM de l'image bruitée
                float ssimFiltered = utils::SSIM(out, original);     // SSIM de l'image filtrée
                // float ssimDiff = ssimFiltered - ssimNoisy;
                float ssimGain = (ssimFiltered / ssimNoisy) - 1.0f;

                if (!compactPrint.Get())
                {
                    std::cout << "================================\n";
                    std::cout << "Image bruitée par " << noiseName << ":\n";
                    std::cout << "PSNR (bruitée): " << psnrNoisy << " dB\n";
                    std::cout << "PSNR (filtrée): " << psnrFiltered << " dB\n";
                    std::cout << "Gain de PSNR : " << psnrGain * 100.0f << " %\n";
                    std::cout << "SSIM (bruitée): " << ssimNoisy << "\n";
                    std::cout << "SSIM (filtrée): " << ssimFiltered << "\n";
                    std::cout << "Gain de SSIM : " << ssimGain * 100.0f << " %\n";
                }
                else
                {
                    std::cout << filterName << " " << psnrNoisy << " " << psnrFiltered << " " << psnrGain * 100.0f
                              << " " << ssimNoisy << " " << ssimFiltered << " " << ssimGain * 100.0f << "\n";
                }
                // if (noiseType == NoiseType::GAUSSIAN)
                //     std::cout << sigma << " " << psnrDiff << " dB\n";

                out.savePNG("../../Ressources/Out/" + originalBaseName + "_" + noiseName + "_denoisedBy" + filterName +
                            ".png");
            }
        }
    }
    else
    {
        // if (compactPrint.Get())
        // {
        //     std::cout << "psnrGain ssimGain\n";
        // }

        if (originalPathParam.Get().empty())
        {
            std::cerr << "Erreur : Le chemin de l'image originale est obligatoire si le bruit n'est pas ajouté."
                      << std::endl;
            return 1;
        }

        Image noisyImage = Image(filepath);
        if (!noisyImage.isLoaded())
        {
            std::cerr << "Erreur : Échec du chargement de l'image bruitée." << std::endl;
            return 1;
        }

        std::string noisyBaseName = utils::basenameWithoutExt(filepath);

        // Charge l'image originale
        Image original(originalPathParam.Get());
        if (!original.isLoaded())
        {
            std::cerr << "Erreur : Échec du chargement de l'image originale." << std::endl;
            return 1;
        }

        Image out = noisyImage;
        for (int i = 0; i < iterNbr; i++)
        {
            out = filter->apply(out); // Applique le filtre
        }

        float psnrNoisy = utils::PSNR(noisyImage, original); // PSNR de l'image bruitée
        float psnrFiltered = utils::PSNR(out, original);     // PSNR de l'image filtrée
        float psnrGain = (psnrFiltered / psnrNoisy) - 1.0f;

        float ssimNoisy = utils::SSIM(noisyImage, original); // SSIM de l'image bruitée
        float ssimFiltered = utils::SSIM(out, original);     // SSIM de l'image filtrée
        float ssimGain = (ssimFiltered / ssimNoisy) - 1.0f;

        if (!compactPrint.Get())
        {
            std::cout << "================================\n";
            std::cout << "PSNR (bruitée): " << psnrNoisy << " dB\n";
            std::cout << "PSNR (filtrée): " << psnrFiltered << " dB\n";
            std::cout << "Gain de PSNR : " << psnrGain * 100.0f << " %\n";
            std::cout << "SSIM (bruitée): " << ssimNoisy << "\n";
            std::cout << "SSIM (filtrée): " << ssimFiltered << "\n";
            std::cout << "Gain de SSIM : " << ssimGain * 100.0f << " %\n";
        }
        else
        {
            std::cout << psnrNoisy << " " << psnrFiltered << " " << psnrGain * 100.0f << " " << ssimNoisy << " "
                      << ssimFiltered << " " << ssimGain * 100.0f << "\n";
        }

        std::string denoisedFilePath = outputPathParam.Get().empty() ? "../../Ressources/Out/" + noisyBaseName +
                                                                           "_denoisedBy" + filterName + ".png"
                                                                     : outputPathParam.Get();
        out.savePNG(denoisedFilePath);

        // std::string nimaScriptCall = "../ffdnet-pytorch/.venv/bin/python3 src/nima.py " + denoisedFilePath;
        // system(nimaScriptCall.c_str());
    }

    delete filter; // Libération de la mémoire du filtre

    return 0; // Succès
}
