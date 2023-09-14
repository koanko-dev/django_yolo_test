# 기 학습되어져서 왼성된 yolo 모델 설정 후 진행
# detect 위해서는 yolov5 패키지가 필요함
# pytorch는 yolov5 패키지를 pip로 제공함 : pip install yolov5

import yolov5
import torch
import os
from PIL import Image as I
from django.conf import settings


def y_detect(img, img_name):
    # model = yolov5.load('y5_model/best.pt')
    # model.conf = 0.5 # confidence
    # model.iou = 0.45 # IOU Threshold 교집합에 해당하는 비율
    # model.multi_label = False # 멀티라벨 여부(다중분류)
    # model.max_det = 1000

    # torch를 이용하는 방법
    # 1. 사용자 학습 모델
    # model = torch.hub.load('ultralytics/yolov5', 'custom', 'y5_model/best.pt') # 두번째 파라미터로 'custom': 직접만들면, 세번째 파라미터로 사용자 학습모델
    # 2. yolov5s 가중치 사용
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # 객체 detect
    img = img
    results = model(img, size=416)
    # print(result.pandas().xyxy[0].value_counts('name')) # 검출된 객체의 이름을 시리즈로 반환

    results.render()  # 출력된 결과의 이미지 사용할 수 있게 변환 (np.array 형식으로 변환)
    static_folder = 'media/'
    inferenced_img_dir = os.path.join(
        static_folder, "inferenced_image")  # 디텍트 이미지 저장경로

    if not os.path.exists(inferenced_img_dir):
        os.makedirs(inferenced_img_dir)

    for img in results.ims:  # np.array 형식
        img_base64 = I.fromarray(img)  # 이미지 형식
        img_base64.save(f"{inferenced_img_dir}/{img_name}")

    res_url = "inferenced_image"+"/" + img_name  # 객체 검출 결과 이미지 저장 경로
    
    return res_url
