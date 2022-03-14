import os
import random
import string

import numpy as np
from PIL import ImageFilter, Image
from torchvision import transforms

import augly
import augly.image as imaugs
import augly.image.functional as F
from augly.image.transforms import BaseTransform
from augly.image.composition import BaseComposition

from typing import Any, Dict, List, Optional, Tuple, Union



SEED = 88

randomRGB = lambda: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

string.ascii_letters + string.digits + string.punctuation
letters = string.ascii_letters + string.digits + string.punctuation
letters = [letter for letter in letters]
randomText = lambda: (''.join(np.random.choice(letters, size = random.randint(2, 10), replace = False)))

IMAGENET_STD = (1,1,1)
IMAGENET_MEAN = (1,1,1)

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
        np.random.shuffle(self.transforms)
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
        
        
        
class OverlayRandomText(imaugs.OverlayText):
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
                              y_pos = random.uniform(0, 0.6),
                              metadata = metadata,
                              bboxes = bboxes,
                              bbox_format = bbox_format)
        
        
        
class RandomSaturation(imaugs.Saturation):
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
        
        
        
class ApplyRandomPILFilter(imaugs.ApplyPILFilter):
    def __init__(self, p: float = 1.0):
        self.filter_types = [ImageFilter.CONTOUR,
                             ImageFilter.EDGE_ENHANCE_MORE,
                             ImageFilter.EMBOSS,
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
        
        
        
class RandomBrightness(imaugs.Brightness):
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
        
        
        
class OverlayOntoRandomBackgroundImage(imaugs.OverlayOntoBackgroundImage):
    def __init__(self, background_image_dir: str, p: float = 1.0,):
        super().__init__(p)
        self.background_images = [os.path.join(background_image_dir, image_path) for image_path in os.path.listdir(background_image_dir)]
        
    def apply_transform(self, image: Image.Image,
                        metadata: Optional[List[Dict[str, Any]]] = None,
                        bboxes: Optional[List[Tuple]] = None,
                        bbox_format: Optional[str] = None,
                        ) -> Image.Image:

        return F.overlay_onto_background_image(image,
                                               background_image = np.random.choice(self.background_images),
                                               opacity = random.uniform(0.8, 1),
                                               overlay_size = random.uniform(0.3, 0.6),
                                               x_pos = random.uniform(0, 0.4),
                                               y_pos = random.uniform(0, 0.4),
                                               scale_bg = False,
                                               metadata = metadata,
                                               bboxes = bboxes,
                                               bbox_format = bbox_format)



class OverlayOntoRandomForegroundImage(imaugs.OverlayOntoBackgroundImage):
    def __init__(self, foreground_image_dir: str, p: float = 1.0,):
        super().__init__(p)
        self.foreground_images = [os.path.join(foreground_image_dir, image_path) for image_path in os.path.listdir(foreground_image_dir)]
        
    def apply_transform(self, image: Image.Image,
                        metadata: Optional[List[Dict[str, Any]]] = None,
                        bboxes: Optional[List[Tuple]] = None,
                        bbox_format: Optional[str] = None,
                        ) -> Image.Image:

        return F.overlay_onto_background_image(np.random.choice(self.foreground_images),
                                               background_image = image,
                                               opacity = random.uniform(0.8, 1),
                                               overlay_size = random.uniform(0.3, 0.6),
                                               x_pos = random.uniform(0, 0.4),
                                               y_pos = random.uniform(0, 0.4),
                                               scale_bg = False,
                                               metadata = metadata,
                                               bboxes = bboxes,
                                               bbox_format = bbox_format)



class RandomShufflePixels(imaugs.ShufflePixels):
    def __init__(self, p: float = 1.0):
        super().__init__(p)

    def apply_transform(self,
                        image: Image.Image,
                        metadata: Optional[List[Dict[str, Any]]] = None,
                        bboxes: Optional[List[Tuple]] = None,
                        bbox_format: Optional[str] = None,
                        ) -> Image.Image:

        return F.shuffle_pixels(image,
                                factor = random.uniform(0.1, 0.3),
                                seed = SEED,
                                metadata = metadata,
                                bboxes = bboxes,
                                bbox_format = bbox_format)
        

class OverlayOntoRandomScreenshot(imaugs.OverlayOntoScreenshot):
    def __init__(self, p: float = 1.0):
        super().__init__(p)
        self.template_filepath = augly.utils.SCREENSHOT_TEMPLATES_DIR
        self.template_bboxes_filepath = augly.utils.BBOXES_PATH

    def apply_transform(self,
                        image: Image.Image,
                        metadata: Optional[List[Dict[str, Any]]] = None,
                        bboxes: Optional[List[Tuple]] = None,
                        bbox_format: Optional[str] = None,
                        ) -> Image.Image:

        return F.overlay_onto_screenshot(image,
                                         template_filepath = [os.path.join(self.template_filepath, f) for f in os.listdir(self.template_filepath) if f.endswith(('png', 'jpg'))],
                                         template_bboxes_filepath = self.template_bboxes_filepath,
                                         max_image_size_pixels = None,
                                         crop_src_to_fit = False,
                                         resize_src_to_match_template = True,
                                         metadata = metadata,
                                         bboxes = bboxes,
                                         bbox_format = bbox_format )
        
        
        
class RandomPadSquare(imaugs.PadSquare):
    def __init__(self, p: float = 1.0):
        super().__init__(p)

    def apply_transform(self,
                        image: Image.Image,
                        metadata: Optional[List[Dict[str, Any]]] = None,
                        bboxes: Optional[List[Tuple]] = None,
                        bbox_format: Optional[str] = None,
                        ) -> Image.Image:

        return F.pad_square(image,
                            color = randomRGB(),
                            metadata = metadata,
                            bboxes = bboxes,
                            bbox_format = bbox_format)
        
        
        
class ConvertRandomColor(imaugs.ConvertColor):
    def __init__(self,
                 mode: Optional[str] = None,
                 matrix: Union[None, 
                               Tuple[float, float, float, float],
                               Tuple[float,float,float,float,float,float,float,float,float,float,float,float],
                               ] = None,
                 dither: Optional[int] = None,
                 palette: int = 0,
                 colors: int = 256,
                 p: float = 1.0):

        super().__init__(p)

    def apply_transform(self,
                        image: Image.Image,
                        metadata: Optional[List[Dict[str, Any]]] = None,
                        bboxes: Optional[List[Tuple]] = None,
                        bbox_format: Optional[str] = None,
                        ) -> Image.Image:
        
        return F.convert_color(image,
                               mode = None,
                               matrix = None,
                               dither = None,
                               palette = 0,
                               colors = random.randint(0, 256),
                               metadata = metadata,
                               bboxes = bboxes,
                               bbox_format = bbox_format)
        

     
        
class RandomChangeAspectRatio(imaugs.ChangeAspectRatio):
    def __init__(self, p: float = 1.0):
        super().__init__(p)

    def apply_transform(self,
                        image: Image.Image,
                        metadata: Optional[List[Dict[str, Any]]] = None,
                        bboxes: Optional[List[Tuple]] = None,
                        bbox_format: Optional[str] = None,
                        ) -> Image.Image:

        return F.change_aspect_ratio(image,
                                     ratio = random.uniform(0.5, 2),
                                     metadata = metadata,
                                     bboxes = bboxes,
                                     bbox_format = bbox_format)
        
        
        

        
        
        
transforms_list = [OverlayRandomStripes(),
                   OverlayRandomEmoji(),
                   RandomPixelization(),
                   EncodingRandomQuality(),
                   OverlayRandomText(),
                   RandomSaturation(),
                   ApplyRandomPILFilter(),
                   RandomBrightness(),
                   OverlayOntoRandomBackgroundImage(),
                   OverlayOntoRandomForegroundImage(),
                   RandomShufflePixels(),
                   OverlayOntoRandomScreenshot(),
                   RandomPadSquare(),
                   ConvertRandomColor(),
                   RandomChangeAspectRatio(),
                   transforms.RandomRotation(p = 1),
                   transforms.GaussianBlur(p = 1),
                   transforms.RandomGrayscale(p = 1),
                   transforms.RandomCrop(p = 1),
                   transforms.RandomPerspective(p = 1),
                   transforms.RandomInvert(p = 1),
                   transforms.RandomPosterize(p = 1),
                   transforms.RandomSolarize(p = 1),
                   transforms.RandomVerticalFlip(p = 1),
                   transforms.RandomHorizontalFlip(p = 1),
                   ]


augs = N_Compositions(transforms_list, n_upper = 6, n_lower = 4)
