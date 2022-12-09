import base64
import cv2
import fnmatch
import mimetypes
import os
import re

ROOT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.join(ROOT_DIR, 'assets', 'images')


def find_image(filename):
    pattern = f'{filename}.*'
    for root, dirs, files in os.walk(BASE_DIR):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name)


def save_image_from_b64(filename, b64_image):
    b64_image = b64_image[b64_image.find(",") + 1:]
    image_binary = base64.b64decode(b64_image)
    image_file_path = os.path.join(BASE_DIR, filename)

    with open(image_file_path, 'wb') as f:
        f.write(image_binary)

    return image_file_path


def convert_to_b64(filepath):
    img = cv2.imread(filepath)
    jpg_img = cv2.imencode('.jpg', img)
    mimetype = mimetypes.guess_type(filepath)[0]
    return f'data:{mimetype};base64,{base64.b64encode(jpg_img[1]).decode("utf-8")}'


def remove_extension(filepath):
    return re.sub(r'\.[^.]*$', '', filepath)
