#include "FFDNetPretrained.hpp"
#include "utils.hpp"

#include <iostream>
#include <string>

Image FFDNetPretrained::apply(const Image &img)
{
    img.savePNG("temp.png");
    const std::string command =
        utils::pythonPath +
        " ../ffdnet-pytorch/test_ffdnet_ipol.py --input "
        "temp.png --output temp_out.png --add_noise False --quiet --model models/net_rgb.pth --pretrained_model";
    system(command.c_str());
    Image out;
    out.load("temp_out.png");
    return out;
}