import glob

import numpy as np
from PIL import Image
import piexif

def luminous_comp():
    files = get_files()
    composite(files)

def get_files():
    return glob.glob('input/*')

def composite(files: list[str]):
    # set first file as base
    res_img = read_file(files[0])
    img_preprocess(res_img)
    exif_dict = piexif.load(res_img.info["exif"])

    # composite
    for idx, file in enumerate(files[1:], start=2):
        cur_img = read_file(file)
        img_preprocess(cur_img)
        res_img = composite_2files(
            res_img,
            cur_img
        )
        print(f'processed {idx} files')

    # save
    exif_bytes = piexif.dump(exif_dict)
    filename = 'result.jpg'
    res_img.save(filename)
    piexif.insert(exif_bytes, filename)

def img_preprocess(img :Image) -> np.array:
    img = img.convert('RGB')
    img = np.array(img)
    img = img.astype(np.float32)
    img = img / 255.0
    return img

def composite_2files(img1 :np.array, img2 :np.array) -> np.array:
    # composite
    res_img = np.maximum(img1, img2)
    res_img = Image.fromarray(res_img)
    return res_img
    
def read_file(file :str) -> Image:
    return Image.open(file)
    
if __name__ == '__main__':
    luminous_comp()
