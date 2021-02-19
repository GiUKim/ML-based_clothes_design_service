# Mask R-CNN for Object Detection and Segmentation (Deepfashion2)

[[원본 깃허브 주소]](https://github.com/matterport/Mask_RCNN)

logs 폴더 생성 후 weight file 다운로드 해서 추가

[[Download Link - trained weight file]](https://drive.google.com/file/d/1U4KDC9Kq1lqJ7JCXQebyHufD0cgmR6nR/view?usp=sharing)


[[Download Link - json files]](https://drive.google.com/drive/folders/1CHPFxcQ6OLFqkHgFsEPybe8kxAy-7-uW?usp=sharing)

---

## 가상환경 세팅

파이썬 3.6.5 버전의 가상환경 생성

conda create -n (envname) python=3.6.5

conda activate (envname)

[참조링크][https://chancoding.tistory.com/86]

---

## 필요한 모듈 설치

### CUDA10.0 Cudnn7.6.4 Windows10

pip install tensorflow-gpu==2.0

pip install keras==2.3.1

pip install numpy==1.19.2

pip install opencv-python

pip install imgaug

pip install Pillow

pip install scikit-image

pip install "h5py<3.0"

pip install Cython

********본인 로컬 경로에 맞게 코드 수정 필요********

balloon.py 38, 79~82 라인 경로 본인 로컬 경로에 맞게 수정 필요

visualize.py 24 라인 경로 본인 로컬 경로에 맞게 수정 필요


cocoapi 설치 및 빌드

* Linux(코랩)
1) https://drive.google.com/drive/folders/1UtmGqkv0F4DcunKIylgzdQ4_biRnraQL?usp=sharing 에서 다운로드 후 DeepLearning 폴더에 압축해제
2) cocoapi_linux -> cocoapi로 디렉토리 이름 변경
3) cocoapi/PythonAPI에서 make 명령어 실행

* Window
1) DeepLearning 폴더에서 git clone https://github.com/philferriere/cocoapi.git
2) cocoapi/PythonAPI에서 python setup.py install 명령어 실행


## 명령어

* [테스팅]
    * python ./samples/balloon/balloon.py splash --weights=./logs/mask_rcnn_deepfashion2_0030.h5 --image=./datasets/test/001186.jpg

* [트레이닝]
    * python ./samples/balloon/balloon.py train --weights=coco

* [평가]
    * python ./samples/balloon/balloon.py evaluate --weights=./logs/mask_rcnn_deepfashion2_0030.h5


