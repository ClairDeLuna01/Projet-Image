#pragma once
#include "Image.hpp"

class Filter
{
  public:
    virtual Image apply(const Image &img) = 0;
};