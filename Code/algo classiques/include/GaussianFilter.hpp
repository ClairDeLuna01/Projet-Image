#pragma once
#include "Filter.hpp"

class GaussianFilter : public Filter
{
  public:
    GaussianFilter(float sigma) : sigma(sigma)
    {
    }

    Image apply(const Image &img) override;

  private:
    float sigma;
};