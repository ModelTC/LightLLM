import torch
import math
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode


def find_closest_aspect_ratio(width, height, min_num=1, max_num=6, image_size=448):
    """
    Find the closest aspect ratio from a list of target ratios to match the given aspect ratio.
    If the difference is the same, use the area to decide the better ratio.
    """
    assert min_num == 1
    log_ratio = math.log(width / height)
    ratio = width * height / (image_size * image_size)
    multiple = min(math.ceil(ratio), max_num)
    if multiple <= 1:
        return [1, 1]
    candidate_split_grids_nums = []
    for i in [multiple - 1, multiple, multiple + 1]:
        if i > max_num:
            continue
        candidate_split_grids_nums.append(i)

    candidate_grids = []
    for split_grids_nums in candidate_split_grids_nums:
        m = 1
        while m <= split_grids_nums:
            if split_grids_nums % m == 0:
                candidate_grids.append([m, split_grids_nums // m])
            m += 1
    best_grid = [1, 1]
    min_error = float("inf")
    for grid in candidate_grids:
        error = abs(log_ratio - math.log(grid[0] / grid[1]))
        if error < min_error:
            best_grid = grid
            min_error = error

    return best_grid


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    """
    Preprocess the image dynamically by finding the closest aspect ratio,
    resizing the image, and splitting it into smaller blocks.
    Optionally add a thumbnail version of the image.
    """
    original_width, original_height = image.size
    target_aspect_ratio = find_closest_aspect_ratio(original_width, original_height, min_num, max_num, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def get_image_patch(orign_width, orign_height, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    """
    Calculate the number of image patches based on the closest aspect ratio
    and the given width and height of the original image.
    """
    target_aspect_ratio = find_closest_aspect_ratio(orign_width, orign_height, min_num, max_num, image_size)
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    if use_thumbnail and blocks != 1:
        blocks += 1
    return blocks


def load_image(image_file, input_size=448, max_num=6):
    """
    Load and preprocess an image file, converting it to RGB mode,
    resizing, normalizing, and optionally adding a thumbnail version.
    """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    image = image_file.convert("RGB")
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
