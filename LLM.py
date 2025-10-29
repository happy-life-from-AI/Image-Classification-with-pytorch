# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 14:42:21 2025

@author: jeonghwan1117
@name  : LLM Based classificaiton with llava
"""

#from ollama import generate  #client 방법이 확실
from ollama import Client
from PIL    import Image
from io     import BytesIO


import base64


## 이미지 파일을 단순히 Base64로 인코딩만 해도 충분
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


def load_image_byte(image_path):
    with open(image_path, 'rb') as t:
        return base64.b64encode(t.read()).decode('utf-8')
        




if __name__ == "__main__":

    image = "./Images/Test.JPG"
    
    image_b64 = load_image_byte(image)
    client    = Client(host='http://localhost:11434')
    prompt    = "이미지는 산업용 검사기에서 검출된 이미지야. 128x128의 이미지가 2장이 붙어 있는 구조이고, 오른쪽은 정상, 왼쪽은 이상이 있는 부분이야. 검은색은 Lead 이고, 흰색은 space인 빈 공간이야. PCB 기판을 투과 조명으로 찍은 사진이야. 한글로 비정상인 부분에 대해 상세하게 설명해줘"
    
    res = client.generate(
        model='llava:13b',
        prompt=prompt,
        images=[image_b64],   # Base64 문자열 배열
        stream= False  
    )
    




