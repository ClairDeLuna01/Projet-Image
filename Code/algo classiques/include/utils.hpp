#pragma once

#include "Image.hpp"
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace utils
{
inline float MSE(const Image &img1, const Image &img2)
{
    if (img1.getWidth() != img2.getWidth() || img1.getHeight() != img2.getHeight() ||
        img1.getChannels() != img2.getChannels())
    {
        std::cerr << "Erreur : Les images doivent avoir les mêmes dimensions et le même nombre de canaux" << std::endl;
        return -1;
    }

    float sum = 0;
    for (int y = 0; y < img1.getHeight(); y++)
    {
        for (int x = 0; x < img1.getWidth(); x++)
        {
            for (int c = 0; c < img1.getChannels(); c++)
            {
                float diff = img1.getClampedPixel(x, y, c) - img2.getClampedPixel(x, y, c);
                sum += diff * diff;
            }
        }
    }

    return sum / (img1.getWidth() * img1.getHeight() * img1.getChannels());
}

inline float PSNR(const Image &img1, const Image &img2)
{
    float mse = MSE(img1, img2);
    if (mse < 0)
    {
        return -1;
    }

    return 10 * log10(255 * 255 / mse);
}

inline std::vector<std::string> getFilesInDirectory(const std::string &directory)
{
    std::vector<std::string> files;

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(directory.c_str())) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            if (ent->d_type == DT_REG)
            {
                files.push_back(ent->d_name);
            }
        }
        closedir(dir);
    }
    else
    {
        std::cerr << "Erreur : Impossible d'ouvrir le dossier " << directory << std::endl;
    }

    return files;
}

inline std::string basename(const std::string &path)
{
    size_t lastSlash = path.find_last_of('/');
    if (lastSlash == std::string::npos)
    {
        return path;
    }
    return path.substr(lastSlash + 1);
}

inline std::string dirname(const std::string &path)
{
    size_t lastSlash = path.find_last_of('/');
    if (lastSlash == std::string::npos)
    {
        return "";
    }
    return path.substr(0, lastSlash);
}

inline bool fileExists(const std::string &path)
{
    std::ifstream f(path.c_str());
    return f.good();
}

inline bool createFolder(const std::string &path)
{
    return mkdir(path.c_str(), 0777) == 0;
}

} // namespace utils
