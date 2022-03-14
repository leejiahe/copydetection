import os
import random
import string

import augly.image as imaugs
import augly.utils as utils
from PIL import ImageFilter, Image
import torchvision
from torchvision import transforms
import numpy as np

import augly
import augly.image.functional as F
import augly.image.utils as imutils
from augly.image.transforms import BaseTransform
from augly.image.composition import BaseComposition

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

string.ascii_letters + string.digits + string.punctuation

randomRGB = lambda: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
letters = string.ascii_letters + string.digits + string.punctuation
letters = [letter for letter in letters]

randomText = lambda: (''.join(np.random.choice(letters, size = random.randint(2, 10), replace = False)))


class N_Compositions(BaseComposition):
    def __init__(self,
                 transforms: List[BaseTransform],
                 n_upper: int,
                 n_lower: int = 1,
                 p: float = 1.0):
        
        super().__init__(transforms, p)
        transform_probs = [t.p for t in transforms]
        probs_sum = sum(transform_probs)
        self.transform_probs = [t / probs_sum for t in transform_probs]
        self.n_upper = n_upper
        self.n_lower = n_lower

    def __call__(self,
                 image: Image.Image,
                 metadata: Optional[List[Dict[str, Any]]] = None,
                 bboxes: Optional[List[Tuple]] = None,
                 bbox_format: Optional[str] = None,
                 ) -> Image.Image:
        
        if random.random() > self.p:
            return image
        
        rand_n = np.random.randint(self.n_lower, self.n_upper)
        transforms = np.random.choice(self.transforms, size = rand_n, replace = False, p = self.transform_probs)
        print(transforms)
        for transform in transforms:
            image = transform(image, force = True, metadata = metadata, bboxes = bboxes, bbox_format = bbox_format)
        return image
    
class OverlayRandomStripes(imaugs.OverlayStripes):
    def __init__(self, p: float = 1.0):
        super().__init__(p)

    def apply_transform(self,
                        image: Image.Image,
                        metadata: Optional[List[Dict[str, Any]]] = None,
                        bboxes: Optional[List[Tuple]] = None,
                        bbox_format: Optional[str] = None,
                        ) -> Image.Image:
        
        return F.overlay_stripes(image,
                                 line_width = random.uniform(0.1, 0.8),
                                 line_color = randomRGB(),
                                 line_angle = random.randrange(-90, 90),
                                 line_density = random.uniform(0.5, 1),
                                 line_type = random.choice(imaugs.utils.SUPPORTED_LINE_TYPES),
                                 line_opacity = random.uniform(0.5, 1),
                                 metadata = metadata,
                                 bboxes = bboxes,
                                 bbox_format = bbox_format)
        
class OverlayRandomEmoji(imaugs.OverlayEmoji):
    def __init__(self, p: float = 1.0,):
        super().__init__(p)
        self.emoji_paths = []
        for folder in os.listdir(augly.utils.EMOJI_DIR):
            files_path = [os.path.join(augly.utils.EMOJI_DIR, folder, file) for file in os.listdir(os.path.join(augly.utils.EMOJI_DIR, folder))]
            self.emoji_paths.extend(files_path)

    def apply_transform(self,
                        image: Image.Image,
                        metadata: Optional[List[Dict[str, Any]]] = None,
                        bboxes: Optional[List[Tuple]] = None,
                        bbox_format: Optional[str] = None) -> Image.Image:
        
        return F.overlay_emoji(image,
                               emoji_path = random.choice(self.emoji_paths),
                               opacity = random.uniform(0.4, 1),
                               emoji_size = random.uniform(0.4, 0.8),
                               x_pos = random.uniform(0, 0.75),
                               y_pos = random.uniform(0, 0.75),
                               metadata = metadata,
                               bboxes = bboxes,
                               bbox_format = bbox_format)
        
class RandomPixelization(imaugs.Pixelization):
    def __init__(self, p: float = 1.0):
        super().__init__(p)

    def apply_transform(self,
                        image: Image.Image,
                        metadata: Optional[List[Dict[str, Any]]] = None,
                        bboxes: Optional[List[Tuple]] = None,
                        bbox_format: Optional[str] = None,
                        ) -> Image.Image:

        return F.pixelization(image,
                              ratio = random.uniform(0.3, 0.8),
                              metadata = metadata,
                              bboxes = bboxes,
                              bbox_format = bbox_format)

class EncodingRandomQuality(imaugs.EncodingQuality):
    def __init__(self, p: float = 1.0):
        super().__init__(p)
 

    def apply_transform(self,
                        image: Image.Image,
                        metadata: Optional[List[Dict[str, Any]]] = None,
                        bboxes: Optional[List[Tuple]] = None,
                        bbox_format: Optional[str] = None,
                        ) -> Image.Image:

        return F.encoding_quality(image,
                                  quality = random.randrange(25, 100),
                                  metadata = metadata,
                                  bboxes = bboxes,
                                  bbox_format = bbox_format)
class OverlayText(BaseTransform):
    def __init__(self,
                 text: List[Union[int, List[int]]] = utils.DEFAULT_TEXT_INDICES,
                 font_file: str = utils.FONT_PATH,
                 font_size: float = 0.15,
                 opacity: float = 1.0,
                 color: Tuple[int, int, int] = utils.RED_RGB_COLOR,
                 x_pos: float = 0.0,
                 y_pos: float = 0.5,
                 p: float = 1.0):

        super().__init__(p)
        self.font_paths = [os.path.join(imaugs.utils.FONTS_DIR, f) for f in os.listdir(imaugs.utils.FONTS_DIR)]
        
    def apply_transform(self,
                        image: Image.Image,
                        metadata: Optional[List[Dict[str, Any]]] = None,
                        bboxes: Optional[List[Tuple]] = None,
                        bbox_format: Optional[str] = None,
                        ) -> Image.Image:

        return F.overlay_text(image,
                              text = randomText(),
                              font_file = random.choice(self.font_paths),
                              font_size = random.uniform(0.05, 0.5),
                              opacity = random.random(),
                              color = randomRGB(),
                              x_pos = random.random(),
                              y_pos = random.random(),
                              metadata = metadata,
                              bboxes = bboxes,
                              bbox_format = bbox_format)
        
        
 class OverlayTextRandom(object):
    def __init__(self):
        pass

    def __call__(self, input_img):
        
        text = []
        text_list = range(1000)
        width = random.randint(5,10)
        for _ in range(random.randint(1,3)):
            text.append(random.sample(text_list, width))
        
        text_size = random.uniform(0.1, 0.4)
            
        aug = imaugs.OverlayText(
            text=text,
            opacity=random.uniform(0.5, 1.0),
            font_size=random.uniform(0.1, 0.4),
            color=random_RGB(),
            x_pos=random.randint(0, 60) * 0.01,
            y_pos=random.randint(0, 60) * 0.01,
        )

        result_img = aug(input_img)

        return result_img