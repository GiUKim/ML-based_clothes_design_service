텐서플로우 1.5

케라스 2.1.5

cocoapi Mask_RCNN 폴더에 설치

**************************
annos, image 폴더 생성
annos에는 json, image에는 사진 파일 저장하기
**************************

트레이닝 명령어(구글드라이브 기준) !python /content/gdrive/MyDrive/Mask_RCNN/samples/balloon/balloon.py train --weights=coco
테스트 명령어 !python Mask_RCNN/Mask_RCNN_DeepFashion2.py splash --weights=Mask_RCNN/mask_rcnn_deepfashion2_0030.h5 --image=Mask_RCNN/DeepFashion2/test/test/image/000004.jpg

경로 수정 필요
