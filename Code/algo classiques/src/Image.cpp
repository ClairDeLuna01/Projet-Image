#include "Image.hpp"
#include "utils.hpp"

bool Image::load(const std::string &filepath)
{
    // Libère d'abord les données existantes, si elles existent
    if (data)
    {
        stbi_image_free(data);
        data = nullptr;
    }
    // Charger l'image
    data = stbi_load(filepath.c_str(), &width, &height, &channels, 0);
    if (!data)
    {
        std::cerr << "Erreur : Impossible de charger l'image " << filepath << std::endl;
        return false;
    }
    // std::cout << "Image chargée : " << filepath << " (" << width << "x" << height << ", " << channels << " canaux)"
    //   << std::endl;
    return true;
}

bool Image::savePNG(const std::string &filepath) const
{
    if (!data)
    {
        std::cerr << "Erreur : Aucune donnée d'image à sauvegarder." << std::endl;
        return false;
    }

    if (!utils::fileExists(utils::dirname(filepath)))
    {
        // std::cout << "Création du dossier : " << utils::dirname(filepath) << std::endl;
        utils::createFolder(utils::dirname(filepath));
    }

    // Sauvegarde de l'image en PNG
    if (stbi_write_png(filepath.c_str(), width, height, channels, data, width * channels))
    {
        // std::cout << "Image sauvegardée en PNG : " << filepath << std::endl;
        return true;
    }
    else
    {
        std::cerr << "Erreur : Échec de la sauvegarde de l'image " << filepath << std::endl;
        return false;
    }
}

Image Image::ApplyConvolution(const ConvolutionKernel &kernel) const
{
    // On crée une image temporaire pour stocker le résultat
    Image out(width, height);

    // On applique le filtre sur chaque pixel
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                float sum = 0;
                float weightSum = 0;

                // On parcourt les pixels du voisinage
                for (int j = -kernel.size / 2; j <= kernel.size / 2; j++)
                {
                    for (int i = -kernel.size / 2; i <= kernel.size / 2; i++)
                    {
                        // On calcule le poids du pixel
                        float weight = kernel(i + kernel.size / 2, j + kernel.size / 2);
                        weightSum += weight;

                        // On ajoute le poids multiplié par la valeur du pixel
                        sum += getClampedPixel(x + i, y + j, c) * weight;
                    }
                }

                // On normalise le résultat
                out.setPixel(x, y, c, utils::clamp(sum / weightSum, 0.0f, 255.0f));
            }
        }
    }

    return out;
}