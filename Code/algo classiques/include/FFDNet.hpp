#pragma once

#include "Filter.hpp"

class FFDNet : public Filter
{
  private:
  public:
    FFDNet()
    {
    }

    Image apply(const Image &img) override;
};