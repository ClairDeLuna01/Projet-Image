#include "GaussianFilter.hpp"
#include <cmath>
#include <iostream>

Image GaussianFilter::apply(const Image &img)
{
    return img.ApplyConvolution(kernel);
}