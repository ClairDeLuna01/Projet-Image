#pragma once

#include "Filter.hpp"

class FFDNetPretrained : public Filter
{
  private:
  public:
    FFDNetPretrained()
    {
    }

    Image apply(const Image &img) override;
};