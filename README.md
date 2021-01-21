# Mask R-CNN for Object Detection and Segmentation

[[원본 깃허브 주소]](https://github.com/matterport/Mask_RCNN)

[[Download Link - trained weight file]](https://drive.google.com/file/d/1SB9HVItbI86-f2rKq3HYAGd9evUzHe7a/view?usp=sharing)

[[Download Link - json files]](https://drive.google.com/drive/folders/1CHPFxcQ6OLFqkHgFsEPybe8kxAy-7-uW?usp=sharing)

---

## CUDA9.0 Windows10 (Training)
## 가상환경 세팅

파이썬 3.6.5 버전의 가상환경 생성

conda create -n (envname) python=3.6.5

conda activate (envname)

[참조링크]: https://chancoding.tistory.com/86

---

## 필요한 모듈 설치

pip install numpy==1.16.1

pip install scikit-image

pip install tensorflow-gpu==1.5

__mrcnn/model.py line20, 21을 주석처리하고 line19 주석 해제한다.__


pip install keras==2.1.5

pip install "h5py<3.0"

pip install --upgrade cython

pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"


---

* [테스팅]
    * python .\samples\balloon\balloon.py splash --weights=.\logs\deepfashion220210108T1354\mask_rcnn_deepfashion2_0040.h5 --image=.\datasets\test\test\image\000071.jpg

* [트레이닝]
    * python .\\samples\\balloon\\balloon.py train --weights=coco


