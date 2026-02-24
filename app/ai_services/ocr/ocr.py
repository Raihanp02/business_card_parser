from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import Image
from typing import Union

from app.ai_services.ocr.postprocess import PostprocessPaddleOCR


ImageInput = Union[str, Image.Image, np.ndarray]

class OCR:
    def __init__(self, postprocessor=PostprocessPaddleOCR()):
        self.model = PaddleOCR(lang="latin", 
                               use_angle_cls=False, 
                               use_gpu=False, 
                               enable_mkldnn=False,
                               det_model_dir="assets/models/whl/det",
                                rec_model_dir="assets/models/whl/rec",
                                cls_model_dir="assets/models/whl/cls",)  # Set 'lang' as needed
        self.postprocessor = postprocessor

    def process(self, image):  
        result, rotated_image = self.parse_raw_text(image)
        sentences = self.postprocessor.postprocess(result, rotated_image)

        return sentences

    def parse_raw_text(self, imgpath):
        img = self._load_image(imgpath)
        result = self.model.ocr(img, cls=False)
        return result, img
    
    def _load_image(self, image: ImageInput) -> np.ndarray:
        """
        Returns image as numpy array (BGR)
        """

        if isinstance(image, str):
            # image path
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Cannot read image from path: {image}")
            return img

        elif isinstance(image, Image.Image):
            # PIL → numpy (RGB → BGR)
            img = np.array(image)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        elif isinstance(image, np.ndarray):
            return image

        else:
            raise TypeError(
                "image must be str (path), PIL.Image.Image, or np.ndarray"
            )