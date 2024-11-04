#pragma once
#include "Image.hpp"

class Filter
{
  public:
    virtual void apply(Image &img) = 0;
};