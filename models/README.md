# 모델 가중치

이 디렉토리에 학습된 YOLO 모델 가중치 파일을 배치합니다.

## 필요한 파일

| 파일명 | 모델 | 용도 | 크기 |
|--------|------|------|------|
| `yolo26n_best.pt` | YOLO26n-OBB | 시설물 탐지 (22클래스) | ~5.8MB |
| `yolov8m_seg_best.pt` | YOLOv8m-seg | 파손 영역 분할 | ~20MB |

## 다운로드

모델 파일은 용량 문제로 Git에 포함되지 않습니다.
아래 링크에서 다운로드 후 이 디렉토리에 배치하세요:

> **[Google Drive 다운로드 링크]** ← 실제 링크로 교체 필요

## 모델 학습 과정

모델 학습에 사용한 코드는 `notebooks/` 디렉토리를 참고하세요.

- `YOLO_Detection.ipynb` — YOLO26n-OBB 학습 과정
- `YOLO_Segmentation.ipynb` — YOLOv8m-seg 학습 과정
- `Feature_Matching_SIFT.ipynb` — SIFT 기반 중복 탐지 실험
