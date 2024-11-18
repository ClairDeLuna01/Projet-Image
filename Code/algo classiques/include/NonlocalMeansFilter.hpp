#pragma once

#include "Filter.hpp"

class NonlocalMeansFilter : public Filter
{
  private:
    float h;
    int patchSize;

  public:
    NonlocalMeansFilter(float h, int patchSize) : h(h), patchSize(patchSize)
    {
    }

    Image apply(const Image &img) override;

    float computeWeight(const Image &img, int x, int y, int i, int j, int c);
};