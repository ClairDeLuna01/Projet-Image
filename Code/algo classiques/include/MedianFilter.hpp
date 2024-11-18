#pragma once

#include "Filter.hpp"

class MedianFilter : public Filter
{
  private:
    int kernelSize;

  public:
    MedianFilter(int kernelSize) : kernelSize(kernelSize)
    {
    }

    Image apply(const Image &img) override;
};