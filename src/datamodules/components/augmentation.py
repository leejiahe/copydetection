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




randomRGB = lambda: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

string.ascii_letters + string.digits + string.punctuation
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
                        bbox_format: Optional[str] = None,
                        ) -> Image.Image:
        
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
                                  quality = random.randrange(5, 50),
                                  metadata = metadata,
                                  bboxes = bboxes,
                                  bbox_format = bbox_format)
        
        
        
class OverlayText(BaseTransform):
    def __init__(self, p: float = 1.0):
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
                              font_size = random.uniform(0.1, 0.4),
                              opacity = random.uniform(0.5, 1),
                              color = randomRGB(),
                              x_pos = random.uniform(0, 0.6),
                              y_pos = random.uniform(0. 0.6),
                              metadata = metadata,
                              bboxes = bboxes,
                              bbox_format = bbox_format)
        
        
        
class Saturation(BaseTransform):
    def __init__(self, p: float = 1.0):
        super().__init__(p)

    def apply_transform(self,
                        image: Image.Image,
                        metadata: Optional[List[Dict[str, Any]]] = None,
                        bboxes: Optional[List[Tuple]] = None,
                        bbox_format: Optional[str] = None,
                        ) -> Image.Image:

        return F.saturation(image,
                            factor = random.uniform(2, 5),
                            metadata = metadata,
                            bboxes = bboxes,
                            bbox_format = bbox_format)
        
        
        
class ApplyPILFilter(BaseTransform):
    def __init__(self, p: float = 1.0):
        self.filter_types = [ImageFilter.BLUR,
                             ImageFilter.CONTOUR,
                             ImageFilter.MaxFilter,
                             ImageFilter.UnsharpMask,
                             ImageFilter.EDGE_ENHANCE,
                             ImageFilter.EDGE_ENHANCE_MORE,
                             ImageFilter.SHARPEN,
                             ImageFilter.SMOOTH_MORE]

        super().__init__(p)

    def apply_transform(self,
                        image: Image.Image,
                        metadata: Optional[List[Dict[str, Any]]] = None,
                        bboxes: Optional[List[Tuple]] = None,
                        bbox_format: Optional[str] = None,
                        ) -> Image.Image:

        return F.apply_pil_filter(image,
                                  filter_type = np.random.choice(self.filter_types),
                                  metadata = metadata,
                                  bboxes = bboxes,
                                  bbox_format = bbox_format)
        
        
        
class Brightness(BaseTransform):
    def __init__(self, p: float = 1.0):
        super().__init__(p)

    def apply_transform(self,
                        image: Image.Image,
                        metadata: Optional[List[Dict[str, Any]]] = None,
                        bboxes: Optional[List[Tuple]] = None,
                        bbox_format: Optional[str] = None,
                        ) -> Image.Image:

        return F.brightness(image,
                            factor = random.random(),
                            metadata = metadata,
                            bboxes = bboxes,
                            bbox_format = bbox_format,)
        
        
        
class PerspectiveTransform(BaseTransform):
    def __init__(self, p: float = 1.0):
        super().__init__(p)


    def apply_transform(self,
                        image: Image.Image,
                        metadata: Optional[List[Dict[str, Any]]] = None,
                        bboxes: Optional[List[Tuple]] = None,
                        bbox_format: Optional[str] = None,
                        ) -> Image.Image:

        return F.perspective_transform(image,
                                       sigma = self.sigma,
                                       dx = self.dx,
                                       dy = self.dy,
                                       seed = self.seed,
                                       metadata = metadata,
                                       bboxes = bboxes,
                                       bbox_format = bbox_format)
