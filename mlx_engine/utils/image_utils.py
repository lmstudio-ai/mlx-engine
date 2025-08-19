import base64
from io import BytesIO
from typing import List
import PIL
import logging

logger = logging.getLogger(__name__)


def convert_to_pil(images_b64: List[str]) -> List[PIL.Image.Image]:
    """Convert a list of base64 strings to PIL Images"""
    return [
        PIL.Image.open(BytesIO(base64.b64decode(img))).convert("RGB")
        for img in images_b64 or []
    ]


def custom_resize(pil_images, max_size=(1000, 1000)):
    """
    Resize and optionally pad a list of PIL images.

    This function resizes images that exceed the specified maximum dimensions,
    maintaining their aspect ratios. If there is more than one image, it also
    pads all images to the same size.

    Args:
        pil_images (list): A list of PIL Image objects to be processed.
        max_size (tuple): Maximum allowed dimensions (width, height) for the images.
                        Defaults to (1000, 1000).

    Returns:
        list: A list of processed PIL Image objects. If there was only one input image,
            it returns the resized image without padding. If there were multiple input
            images, it returns padded images of uniform size.

    Side effects:
        Writes progress and status messages to sys.stderr.
    """
    resized_images = []
    max_width, max_height = 0, 0

    for i, img in enumerate(pil_images):
        original_size = (img.width, img.height)
        aspect_ratio = img.width / img.height

        logger.info(
            f"Image {i + 1}: Original size {original_size}",
        )

        if img.width > max_size[0] or img.height > max_size[1]:
            if img.width > img.height:
                new_width = max_size[0]
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = max_size[1]
                new_width = int(new_height * aspect_ratio)
            img = img.resize((new_width, new_height), PIL.Image.LANCZOS)
            logger.info(
                f"Image {i + 1}: Resized to {img.width}x{img.height}\n",
            )
        else:
            logger.info(
                f"Image {i + 1}: No resize needed\n",
            )

        max_width = max(max_width, img.width)
        max_height = max(max_height, img.height)

        resized_images.append(img)

    if len(pil_images) > 1:
        logger.info(
            f"[mlx-engine] Maximum dimensions: {max_width}x{max_height}. "
            f"Adding padding so that all images are the same size.\n",
        )

        final_images = []
        for i, img in enumerate(resized_images):
            new_img = PIL.Image.new("RGB", (max_width, max_height), (0, 0, 0))
            paste_x = (max_width - img.width) // 2
            paste_y = (max_height - img.height) // 2
            new_img.paste(img, (paste_x, paste_y))
            final_images.append(new_img)
        return final_images
    else:
        return resized_images
