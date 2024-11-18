#pragma once

#include "./extern/stb_image.h"
#include "./extern/stb_image_write.h"

#include <cstdlib>
#include <iostream>
#include <string>

struct ConvolutionKernel
{
    int size;
    float *data;

    ConvolutionKernel(int size, float *data) : size(size), data(data)
    {
    }

    ConvolutionKernel(int size) : size(size)
    {
        data = new float[size * size];
    }

    ConvolutionKernel() : size(0), data(nullptr)
    {
    }

    ConvolutionKernel(const ConvolutionKernel &other) : size(other.size)
    {
        data = new float[size * size];
        std::copy(other.data, other.data + size * size, data);
    }

    ~ConvolutionKernel()
    {
        if (data)
        {
            delete[] data;
        }
    }

    float &operator()(int x, int y)
    {
        return data[y * size + x];
    }

    float operator()(int x, int y) const
    {
        return data[y * size + x];
    }

    float *operator[](int y)
    {
        return data + y * size;
    }
};

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
        // utilisation de malloc car les données de l'image sont utilisées par stb_image
        data = (unsigned char *)malloc(width * height * channels);
    }

    // Constructeur de copie
    Image(const Image &other) : width(other.width), height(other.height), channels(other.channels)
    {
        data = (unsigned char *)malloc(width * height * channels);
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

    // Opérateur d'affectation
    Image &operator=(const Image &other)
    {
        if (this != &other)
        { // Vérification d'auto-affectation
            // Libérer la mémoire existante
            if (data)
            {
                free(data);
            }

            // Copier les attributs de base
            width = other.width;
            height = other.height;
            channels = other.channels;

            // Allouer de la mémoire et copier les données
            data = (unsigned char *)malloc(width * height * channels);
            std::copy(other.data, other.data + width * height * channels, data);
        }
        return *this;
    }

    // Fonction pour charger une image
    bool load(const std::string &filepath);

    // Fonction pour sauvegarder une image au format PNG
    bool savePNG(const std::string &filepath) const;

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

    Image ApplyConvolution(const ConvolutionKernel &kernel) const;

    // Surcharge de l'opérateur () pour accéder à un pixel
    unsigned char operator()(int x, int y, int c = 0) const
    {
        return data[(y * width + x) * channels + c];
    }

    // Surcharge de l'opérateur () pour modifier un pixel
    unsigned char &operator()(int x, int y, int c = 0)
    {
        return data[(y * width + x) * channels + c];
    }
};
