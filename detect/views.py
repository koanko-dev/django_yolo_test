from django.shortcuts import render
import os
from django.core.files.storage import FileSystemStorage
from uuid import uuid4  # 고유번호 생성하는 라이브러리

from . import yolov5_detect


def index(request):
    return render(request, 'detect/img_up_res.html')


def rename_imagefile_to_uuid(filename):  # 고유번호 이용해서 이미지 파일명 변경하는 함수
    ext = filename.split('.')[-1]  # 확장자 분리 후, 확장자만 저장
    uuid = uuid4().hex
    filename = '{}.{}'.format(uuid, ext)
    return filename


def detect(request):
    img = request.FILES.get('images')
    fs = FileSystemStorage()  # 파일저장소 접근 객체

    # 전송된 파일명 변경해서 저장(서버에서 사용할 유일한 파일명 생성)
    file_name = rename_imagefile_to_uuid(img.name)
    img_up_url = fs.save(file_name, img)  # media 디렉토리에 저장됨
    print(img_up_url)

    # 저장한 이미지 가져와서 객체 검출 함수를 호출해 사용(yolov5_detect.py)
    img = 'media/'+img_up_url
    res_url = yolov5_detect.y_detect(img, img_up_url)
    
    print(res_url)

    return render(request, 'detect/result.html', {
        'image': res_url
    })
