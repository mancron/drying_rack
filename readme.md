# 스마트 건조대 AI 예측 시스템 (Smart Drying Rack)

## 프로젝트 개요

Firebase Realtime Database에 수집된 스마트 건조대의 센서 데이터(주변 온도/습도, 조도, 4개의 개별 옷 습도)를 바탕으로 XGBoost 모델을 학습하여, 각 센서별 남은 건조 시간을 실시간으로 예측하는 사물인터넷(IoT) 시스템입니다. 라즈베리파이 환경에서 동작하며 예측 결과를 Firebase와 MQTT를 통해 외부로 전송합니다.

## 주요 기능

* **센서별 독립 AI 모델 학습 (`drying_time_predictor.py`)**: 4개의 수분 센서에 대해 각각 독립적인 XGBoost 회귀 모델을 학습합니다. 건조 완료 시점을 자동 탐지하고, 습도 변화량 및 추세 등 파생 변수를 생성하여 정밀도를 높입니다.
* **실시간 예측 및 IoT 통신 (`raspberry.py`)**: 라즈베리파이에서 백그라운드로 동작하며 Firebase의 예측 명령을 리스닝합니다. 명령 수신 시 최근 센서 데이터를 불러와 남은 건조 시간을 예측한 뒤, 결과를 Firebase와 MQTT 브로커로 발행(Publish)합니다.
* **데이터 시각화 (`data_visualizer.py`, `heatmap_generator.py`)**: 과거 건조 세션의 원본 센서 데이터와 모델 학습용으로 전처리된 피처 데이터를 그래프 및 상관관계 히트맵으로 시각화하여 분석할 수 있습니다.
* **데이터베이스 연동 (`firebase_manager.py`)**: `firebase-admin`을 활용해 Realtime Database의 경로 데이터를 순차적으로 병합하고 Pandas DataFrame으로 변환합니다.

## 기술 스택

* **언어**: Python
* **머신러닝 및 데이터**: XGBoost, Scikit-learn, Pandas, NumPy
* **시각화**: Matplotlib, Seaborn
* **인프라 및 네트워크**: Firebase Realtime Database, MQTT (`paho-mqtt`)

## 설치 및 셋업

### 1. 필수 패키지 설치

터미널에서 아래 명령어를 실행하여 필요 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt

```

### 2. Firebase 인증 키 설정

* 프로젝트 루트 경로에 Firebase 서비스 계정 키 파일인 `firebase.json`을 위치시켜야 합니다.
* DB URL 변경이 필요할 경우 각 코드 파일의 `DATABASE_URL` 변수를 수정합니다.

## 실행 방법

### 단계 1: 모델 학습 및 생성

가장 먼저 과거 데이터를 기반으로 AI 모델을 학습시켜야 합니다. 실행이 완료되면 `all_sensors_bundle.pkl` 등의 모델 파일이 생성됩니다.

```bash
python drying_time_predictor.py

```

### 단계 2: 실시간 예측 에이전트 구동 (라즈베리파이)

학습된 모델 번들을 로드한 상태로 대기하며, Firebase `/drying-rack/command` 경로에 요청이 들어오면 실시간 건조 시간을 예측하여 반환합니다.

```bash
python raspberry.py

```

### 참고: 데이터 시각화 툴

수집된 전체 건조 세션의 흐름이나 전처리된 피처의 상태를 눈으로 확인하고 싶을 때 실행합니다.

```bash
python data_visualizer.py

```
