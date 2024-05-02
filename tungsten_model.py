# flake8: noqa
import os
import tempfile
import warnings
import uuid
warnings.filterwarnings("ignore", module="torchvision", category=UserWarning)

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from pathlib import Path
from PIL import Image as PILImage
from tungstenkit import BaseIO, Field, Image, Option, define_model

from gfpgan import GFPGANer
from realesrgan.utils import RealESRGANer


class Input(BaseIO):
    img: Image = Field(description="Input image")
    version: str = Option(
        description="RealESRGAN version",
        choices=[
            "General - RealESRGANplus",
            "General - v3",
            "Anime - anime6B",
            "AnimeVideo - v3",
        ],
        default="General - v3",
    )
    scale: float = Option(description="Rescaling factor", default=2, ge=1, le=4)
    face_enhance: bool = Option(
        description="Enhance faces with GFPGAN. Note that it does not work for anime images",
        default=False,
    )
    tile: int = Option(
        description="Tile size. Default is 0, that is no tile. When encountering the out-of-GPU-memory issue, please specify it, e.g., 400 or 200",
        default=0,
        ge=0,
    )


class Output(BaseIO):
    enhanced: Image


@define_model(
    input=Input,
    output=Output,
    batch_size=1,
    gpu=True,
    gpu_mem_gb=16,
    system_packages=["libgl1-mesa-glx", "libglib2.0-0"],
    python_packages=[
        "torch==1.7.1",
        "torchvision==0.8.2",
        "numpy==1.21.1",
        "lmdb==1.2.1",
        "opencv-python==4.5.3.56",
        "PyYAML==5.4.1",
        "tqdm==4.62.2",
        "yapf==0.31.0",
        "basicsr==1.4.2",
        "facexlib==0.2.5",
        "gfpgan==1.3.8",
        "basicsr==1.4.2",
        "tqdm",
    ],
)
class RealESRGAN:
    def choose_model(self, scale: float, version: str, tile: int = 0):
        half = True if torch.cuda.is_available() else False
        if version == "General - RealESRGANplus":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            model_path = "weights/RealESRGAN_x4plus.pth"
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=half,
            )
        elif version == "General - v3":
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=4,
                act_type="prelu",
            )
            model_path = "weights/realesr-general-x4v3.pth"
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=half,
            )
        elif version == "Anime - anime6B":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,
                num_grow_ch=32,
                scale=4,
            )
            model_path = "weights/RealESRGAN_x4plus_anime_6B.pth"
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=half,
            )
        elif version == "AnimeVideo - v3":
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=16,
                upscale=4,
                act_type="prelu",
            )
            model_path = "weights/realesr-animevideov3.pth"
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=half,
            )

        self.face_enhancer = GFPGANer(
            model_path="weights/GFPGANv1.4.pth",
            upscale=scale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=self.upsampler,
        )

    def predict(self, inputs: list[Input]) -> list[Output]:
        input = inputs[0]

        img = input.img.path
        tile = input.tile
        version = input.version
        scale = input.scale
        face_enhance = input.face_enhance

        if tile <= 100 or tile is None:
            tile = 0
        print(
            f"img: {img.name}. version: {version}. scale: {scale}. face_enhance: {face_enhance}. tile: {tile}."
        )

        img = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = "RGBA"
        elif len(img.shape) == 2:
            img_mode = None
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_mode = None

        h, w = img.shape[0:2]
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        self.choose_model(scale, version, tile)

        try:
            if face_enhance:
                _, _, output = self.face_enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True
                )
            else:
                output, _ = self.upsampler.enhance(img, outscale=scale)
        except RuntimeError as error:
            print("Error", error)
            print(
                'If you encounter CUDA out of memory, try to set "tile" to a smaller size, e.g., 400.'
            )

        fname = f"{uuid.uuid4().hex[:8]}.png"
        out_path = Path(tempfile.mkdtemp()) / fname
        cv2.imwrite(str(out_path), output)
        output = Output(enhanced=Image.from_path(out_path))
        return [output]
