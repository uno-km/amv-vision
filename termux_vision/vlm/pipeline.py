import os
import tempfile
from typing import Union, Dict, Any
import numpy as np
from ..io.loader import load_image, save_image
from ..transforms.functional import resize
from .model_hub import download_vlm_model
from .engine import ZeroFlickerEngine

class VLM:
    def __init__(self, engine: ZeroFlickerEngine):
        self.engine = engine

    def describe(
        self,
        image: Union[str, np.ndarray],
        prompt: str = "이 사진 속 인물의 표정, 옷차림, 자세, 그리고 배경 환경을 한국어로 자세히 설명해줘.",
        max_tokens: int = 200,
        resolution: int = 384
    ) -> Dict[str, Any]:
        """
        High-level VLM image description API.
        Accepts image path or NumPy array, safely resizes to target resolution (default 384x384),
        and generates descriptive natural language text.
        """
        temp_img_path = None
        if isinstance(image, str):
            if os.path.exists(image):
                # Preprocess & downsample for memory safety
                raw_img = load_image(image)
                res_img = resize(raw_img, (resolution, resolution))
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    temp_img_path = tmp.name
                save_image(res_img, temp_img_path, quality=90)
                input_file = temp_img_path
            else:
                raise FileNotFoundError(f"Image not found: {image}")
        elif isinstance(image, np.ndarray):
            res_img = resize(image, (resolution, resolution))
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_img_path = tmp.name
            save_image(res_img, temp_img_path, quality=90)
            input_file = temp_img_path
        else:
            raise ValueError("Input image must be a file path string or NumPy ndarray.")

        try:
            return self.engine.generate(input_file, prompt=prompt, max_tokens=max_tokens)
        finally:
            if temp_img_path and os.path.exists(temp_img_path):
                os.remove(temp_img_path)

    def ask(self, image: Union[str, np.ndarray], question: str) -> str:
        res = self.describe(image, prompt=question)
        return res.get("text", "")

def load_vlm(
    model_name: str = "qwen2-vl-2b",
    threads: int = 4,
    eco_mode: bool = True,
    use_vulkan: bool = False
) -> VLM:
    """
    Factory function to download/load an On-Device VLM instance.
    """
    text_path, vision_path = download_vlm_model(model_name)
    engine = ZeroFlickerEngine(
        text_model_path=text_path,
        vision_model_path=vision_path,
        threads=threads,
        eco_mode=eco_mode,
        use_vulkan=use_vulkan
    )
    return VLM(engine)
