#pragma once

#include "Filter.hpp"

class BilateralFilter : public Filter
{
  private:
    int kernelSize;
    float sigmaD;
    float sigmaR;

  public:
    BilateralFilter(int kernelSize, float sigmaD, float sigmaR) : kernelSize(kernelSize), sigmaD(sigmaD), sigmaR(sigmaR)
    {
    }

    Image apply(const Image &img) override;
};