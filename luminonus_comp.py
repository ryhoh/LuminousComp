import glob

import numpy as np
from PIL import Image
import piexif

def luminous_comp():
    files = get_files()
    composite(files, fade=False)

def get_files():
    return sorted(glob.glob('input/*'))

def composite(files: list[str], fade: bool = False):
    # set black file as base
    first_img = read_file(files[0])
    exif_dict = piexif.load(first_img.info["exif"])
    first_img = img_preprocess(first_img)
    res_array = np.zeros_like(first_img)

    # composite
    for idx, file in enumerate(files, start=1):
        cur_img = read_file(file)
        cur_array = img_preprocess(cur_img)

        if fade:
            # fade enabled
            img2_weight = fade_weight_sin_with_25_percent(idx, len(files) + 1)
        else:
            # fade disabled
            img2_weight = 1.0
            
        res_array = composite_2files(
            res_array,
            cur_array,
            img2_weight
        )
        print(f'processed {idx} files {img2_weight}')

    # save
    res_img = img_postprocess(res_array)
    exif_bytes = piexif.dump(exif_dict)
    filename = 'output/result.jpg'
    res_img.save(filename)
    piexif.insert(exif_bytes, filename)

def img_preprocess(img :Image) -> np.array:
    img = img.convert('RGB')
    img = np.array(img)
    img = img.astype(np.float32)
    img = img / 255.0
    return img

# fade enabled for first 50 % of files, and increase weight linearly
def fade_weight_linear(idx: int, total: int) -> float:
    return 1.0 - abs((idx / total) - 0.5) * 2

def fade_weight_cos(idx: int, total: int) -> float:
    return 1.0 - abs(np.cos((idx / total) * np.pi))

def fade_weight_sin(idx: int, total: int) -> float:
    return np.sin((idx / total) * np.pi)

def fade_weight_sin_with_25_percent(idx: int, total: int) -> float:
    if idx < total * 0.25:
        return np.sin((idx / (total * 0.25)) * np.pi / 2)
    elif idx < total * 0.75:
        return 1.0
    else:
        return 1.0 - np.sin(((idx - (total * 0.75)) / (total * 0.25)) * np.pi / 2)
    

def composite_2files(img1 :np.array, img2 :np.array, img2_weight: float = 1.0) -> np.array:
    # check weight
    if img2_weight < 0.0 or img2_weight > 1.0:
        raise ValueError('img2_weight must be between 0.0 and 1.0')

    # composite
    res_img = np.maximum(img1, img2 * img2_weight)
    return res_img

def img_postprocess(img :np.array) -> Image:
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img
    
def read_file(file :str) -> Image:
    return Image.open(file)
    
if __name__ == '__main__':
    luminous_comp()
