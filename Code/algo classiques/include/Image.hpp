#pragma once

#include "./extern/stb_image.h"
#include "./extern/stb_image_write.h"
#include <iostream>
#include <string>

class Image
{
  private:
    unsigned char *data; // pointeur sur les données de l'image
    int width;           // largeur de l'image
    int height;          // hauteur de l'image
    int channels;        // nombre de canaux (par ex., 3 pour RGB, 4 pour RGBA)

  public:
    // Constructeur par défaut
    Image() : data(nullptr), width(0), height(0), channels(0)
    {
    }

    // Constructeur pour charger une image
    Image(const std::string &filepath) : data(nullptr), width(0), height(0), channels(0)
    {
        load(filepath);
    }

    // Constructeur pour créer une image vide
    Image(int width, int height, int channels = 3) : width(width), height(height), channels(channels)
    {
        data = new unsigned char[width * height * channels]();
    }

    // Constructeur de copie
    Image(const Image &other) : width(other.width), height(other.height), channels(other.channels)
    {
        data = new unsigned char[width * height * channels];
        std::copy(other.data, other.data + width * height * channels, data);
    }

    // Destructeur pour libérer la mémoire
    ~Image()
    {
        if (data)
        {
            stbi_image_free(data);
        }
    }

    // Fonction pour charger une image
    bool load(const std::string &filepath)
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
        std::cout << "Image chargée : " << filepath << " (" << width << "x" << height << ", " << channels << " canaux)"
                  << std::endl;
        return true;
    }

    // Fonction pour sauvegarder une image au format PNG
    bool savePNG(const std::string &filepath) const
    {
        if (!data)
        {
            std::cerr << "Erreur : Aucune donnée d'image à sauvegarder." << std::endl;
            return false;
        }
        // Sauvegarde de l'image en PNG
        if (stbi_write_png(filepath.c_str(), width, height, channels, data, width * channels))
        {
            std::cout << "Image sauvegardée en PNG : " << filepath << std::endl;
            return true;
        }
        else
        {
            std::cerr << "Erreur : Échec de la sauvegarde de l'image " << filepath << std::endl;
            return false;
        }
    }

    // Fonction pour obtenir la largeur de l'image
    int getWidth() const
    {
        return width;
    }

    // Fonction pour obtenir la hauteur de l'image
    int getHeight() const
    {
        return height;
    }

    // Fonction pour obtenir le nombre de canaux
    int getChannels() const
    {
        return channels;
    }

    // Vérifie si l'image a été chargée
    bool isLoaded() const
    {
        return data != nullptr;
    }

    // Fonction pour obtenir la valeur d'un pixel (avec gestion des bords)
    unsigned char getClampedPixel(int x, int y, int c = 0) const
    {
        x = std::max(0, std::min(x, width - 1));
        y = std::max(0, std::min(y, height - 1));
        return data[(y * width + x) * channels + c];
    }

    // Fonction pour définir la valeur d'un pixel
    void setPixel(int x, int y, int c, unsigned char value)
    {
        data[(y * width + x) * channels + c] = value;
    }

    // Affiche des informations sur l'image
    void printInfo() const
    {
        if (data)
        {
            std::cout << "Image info : " << width << "x" << height << ", " << channels << " canaux" << std::endl;
        }
        else
        {
            std::cout << "Aucune image chargée" << std::endl;
        }
    }
};
