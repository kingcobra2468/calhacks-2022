import base64
import os

from config import ROOT_DIR

def save_image_base64(contents, id):
    img_path = os.path.join(ROOT_DIR, 'static/images/', f'{id}.png')
    with open(img_path , 'wb') as fh:
        fh.write(base64.b64decode(contents))

    return os.path.join('/static/', f'{id}.png')
