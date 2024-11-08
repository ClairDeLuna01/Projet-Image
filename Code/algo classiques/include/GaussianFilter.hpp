#pragma once
#include "cmath"

#include "Filter.hpp"

class GaussianFilter : public Filter
{
  private:
    float sigma;
    ConvolutionKernel kernel;

  public:
    GaussianFilter(float sigma) : sigma(sigma), kernel(1 + 2 * ceil(2 * sigma))
    {
        // On récupère la taille du noyau qu'on a calculé au dessus a partir de sigma
        int kernelSize = kernel.size;

        // On remplit le noyau avec les valeurs de la gaussienne
        float sum = 0;
        for (int j = -kernelSize / 2; j <= kernelSize / 2; j++)
        {
            for (int i = -kernelSize / 2; i <= kernelSize / 2; i++)
            {
                float weight = exp(-(i * i + j * j) / (2 * sigma * sigma));
                // std::cout << "(i, j) = (" << i << ", " << j << ") => weight = " << weight << std::endl;
                kernel(i + kernelSize / 2, j + kernelSize / 2) = weight;
                sum += weight;
            }
        }

        // On normalise le noyau
        for (int j = -kernelSize / 2; j <= kernelSize / 2; j++)
        {
            for (int i = -kernelSize / 2; i <= kernelSize / 2; i++)
            {
                kernel(i + kernelSize / 2, j + kernelSize / 2) /= sum;
            }
        }
    }

    ~GaussianFilter()
    {
    }

    Image apply(const Image &img) override;
};