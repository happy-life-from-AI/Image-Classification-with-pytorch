# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 14:42:21 2025

@author: jeonghwan1117
@name  : LLM Based classificaiton with llava
"""

from ollama import generate

from PIL import Image
from io  import BytesIO


def load_image(image_path):
    '''
    Parameters
    ----------
    image_path : Image Path

    Returns
    -------
    image_bytes

    '''
    with Image.open(image_path) as img:
        with BytesIO() as bf:
            img.save(bf, format = 'PNG')
            img_b = bf.getvalue()
            

    return img_b




if __name__ == "__main__":

    image = "./Images/Test.JPG"




