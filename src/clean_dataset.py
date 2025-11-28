# clean_dataset.py
import os
from PIL import Image

def clean_folder(root, min_size=256):
    count = 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            path = os.path.join(dirpath, f)
            try:
                img = Image.open(path)
                w, h = img.size
                if w < min_size or h < min_size:
                    os.remove(path)
                    continue
            except:
                os.remove(path)
                continue
            count += 1
    print("Valid images:", count)

if __name__ == "__main__":
    # clean_folder("data/content/val2017", 256)
    # clean_folder("data/style/Artworks", 256)
    clean_folder("data/content/train2014", 256)
