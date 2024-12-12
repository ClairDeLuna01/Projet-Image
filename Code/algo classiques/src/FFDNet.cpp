#include "FFDNet.hpp"

#include <iostream>
#include <string>

Image FFDNet::apply(const Image &img)
{
    img.savePNG("temp.png");
    const std::string command = "python3.9 ../ffdnet-pytorch/test_ffdnet_ipol.py --input "
                                "temp.png --output temp_out.png --add_noise False --quiet --model models/net_2.pth";
    system(command.c_str());
    Image out;
    out.load("temp_out.png");
    return out;
}