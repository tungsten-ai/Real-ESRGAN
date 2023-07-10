#!/bin/bash

set -eu

if [ ! -f weights/realesr-general-x4v3.pth ]
then
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P ./weights
fi

if [ ! -f weights/GFPGANv1.4.pth ]
then
    wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P ./weights
fi

if [ ! -f weights/RealESRGAN_x4plus.pth ]
then
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ./weights
fi

if [ ! -f weights/RealESRGAN_x4plus_anime_6B.pth ]
then
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P ./weights
fi

if [ ! -f weights/realesr-animevideov3.pth ]
then
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth -P ./weights
fi

if [ ! -f gfpgan/weights/detection_Resnet50_Final.pth ]
then
    wget https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -P ./gfpgan/weights
fi

if [ ! -f gfpgan/weights/parsing_parsenet.pth ]
then
    wget https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth -P ./gfpgan/weights
fi