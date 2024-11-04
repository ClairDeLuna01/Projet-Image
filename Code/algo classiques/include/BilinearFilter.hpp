#pragma once

#include "Filter.hpp"

class BilinearFilter : public Filter
{
  public:
    BilinearFilter()
    {
    }

    Image apply(const Image &img) override;

  private:
};