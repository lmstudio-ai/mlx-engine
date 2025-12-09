import base64
import logging
from io import BytesIO
from typing import List

import PIL

logger = logging.getLogger(__name__)


def convert_to_pil(images_b64: List[str]) -> list[PIL.Image.Image]:
    """Convert a list of base64 strings to PIL Images"""
    return [
        PIL.Image.open(BytesIO(base64.b64decode(img))).convert("RGB")
        for img in images_b64 or []
    ]


def custom_resize(
    pil_images: list[PIL.Image.Image],
    *,
    max_size: tuple[int, int] | None,
    should_pad: bool = True,
):
    """
    Resize and optionally pad a list of PIL images.

    This function resizes images that exceed the specified maximum dimensions,
    maintaining their aspect ratios. If there is more than one image, it also
    pads all images to the same size.

    Args:
        pil_images (list): A list of PIL Image objects to be processed.
        max_size (tuple): Maximum allowed dimensions (width, height) for the images.
                        If None, no resizing is performed.
        should_pad (bool): Whether to pad the images to the same size.
                        Defaults to True.

    Returns:
        list: A list of processed PIL Image objects. If there was only one input image,
            it returns the resized image without padding. If there were multiple input
            images, it returns padded images of uniform size.

    Side effects:
        Writes progress and status messages to sys.stderr.
    """
    # Validate max_size parameter
    if max_size is not None:
        if not isinstance(max_size, tuple) or len(max_size) != 2:
            raise ValueError(
                "max_size must be a tuple of (width, height), e.g., (1024, 1024)"
            )
        if not all(isinstance(dim, int) and dim > 0 for dim in max_size):
            raise ValueError("max_size dimensions must be positive integers")

    resized_images = []
    max_width, max_height = 0, 0

    for i, img in enumerate(pil_images):
        original_width, original_height = img.width, img.height
        aspect_ratio = img.width / img.height

        if max_size is not None and (
            img.width > max_size[0] or img.height > max_size[1]
        ):
            if img.width > img.height:
                new_width = max_size[0]
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = max_size[1]
                new_width = int(new_height * aspect_ratio)
            img = img.resize((new_width, new_height), PIL.Image.LANCZOS)
            logger.info(
                f"Image {i + 1}: Resized from {original_width}x{original_height} to {img.width}x{img.height}\n",
            )

        max_width = max(max_width, img.width)
        max_height = max(max_height, img.height)

        resized_images.append(img)

    if len(pil_images) > 1 and should_pad:
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
