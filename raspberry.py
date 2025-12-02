import time
import threading
import json
import joblib
import pandas as pd
import paho.mqtt.client as mqtt
from firebase_admin import db

# 기존 파일들에서 필요한 클래스와 함수 임포트
from firebase_manager import RealtimeDatabaseManager
from drying_time_predictor import make_features_for_prediction

# --- 설정 (Configuration) ---
FIREBASE_KEY_PATH = "firebase.json"
DATABASE_URL = "https://smart-drying-rack-fe271-default-rtdb.firebaseio.com/"
BASE_DATA_PATH = "drying-rack"  # 데이터를 가져올 경로
COMMAND_PATH = "/drying-rack/command"  # 앱에서 명령을 보낼 경로 (예: "start_prediction")

MQTT_BROKER = "broker.hivemq.com"  # 사용할 MQTT 브로커 주소 (변경 가능)
MQTT_PORT = 1883
MQTT_TOPIC_RESULT = "drying_rack/prediction_result"  # 결과를 보낼 토픽
MQTT_TOPIC_STATUS = "drying_rack/status"  # 상태(처리중/대기중)를 보낼 토픽

# 학습 때 사용한 피처 순서 (매우 중요: 모델 학습시와 동일해야 함)
MODEL_FEATURES = [
    'ambient_temp', 'ambient_humidity', 'light_lux_avg',
    'cloth_humidity', 'Δhumidity', 'Δillumination', 'humidity_trend'
]

# --- 전역 변수 ---
is_processing = False  # 중복 실행 방지 플래그
model = None
scaler = None
rtdb_manager = None
mqtt_client = None


def load_ai_models():
    """저장된 AI 모델과 스케일러 불러오기"""
    global model, scaler
    try:
        model = joblib.load('drying_model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("✅ 모델과 스케일러 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print("   (먼저 drying_time_predictor.py를 실행해 모델을 생성해주세요)")
        exit()


def get_latest_data_for_prediction():
    """RTDB에서 최근 데이터를 가져와 예측용 피처로 변환"""
    try:
        # 최근 데이터를 충분히 가져옴 (이동 평균 계산 등을 위해 10개 정도)
        # fetch_sequential_paths_as_dataframe은 전체를 가져오므로,
        # 실제 운영시에는 마지막 노드만 가져오는 최적화가 필요할 수 있으나,
        # 현재 구조상 전체를 가져와서 tail()을 씁니다.
        df = rtdb_manager.fetch_sequential_paths_as_dataframe(BASE_DATA_PATH)

        if df.empty or len(df) < 3:
            print("⚠ 데이터가 부족하여 예측할 수 없습니다 (최소 3개 필요).")
            return None

        # 마지막 3개 데이터만 추출하여 피처 생성
        current_data_slice = df.tail(3).copy().reset_index(drop=True)

        # drying_time_predictor.py에 있는 함수 재사용
        features = make_features_for_prediction(current_data_slice, MODEL_FEATURES)
        return features

    except Exception as e:
        print(f"❌ 데이터 조회/변환 중 오류: {e}")
        return None


def prediction_worker(command_data):
    """별도 쓰레드에서 실행될 예측 작업"""
    global is_processing

    print(f"▶ [작업 시작] 명령: {command_data}")
    mqtt_client.publish(MQTT_TOPIC_STATUS, "BUSY")  # 상태 알림

    try:
        # 1. 최신 데이터 준비
        features = get_latest_data_for_prediction()

        if features is not None:
            # 2. AI 예측 수행
            scaled_features = scaler.transform(features)
            predicted_time = model.predict(scaled_features)[0]
            predicted_time = max(0, predicted_time)  # 음수 방지

            result_msg = {
                "predicted_minutes": round(predicted_time, 1),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            db.reference("/drying-rack/result").set(result_msg)
            print("✅ 파이어베이스에 결과 저장 완료 (/drying-rack/result)")

            # 3. MQTT 전송
            payload = json.dumps(result_msg, ensure_ascii=False)
            mqtt_client.publish(MQTT_TOPIC_RESULT, payload)
            print(f"◀ [예측 성공] 남은 시간: {predicted_time:.1f}분 -> MQTT 전송 완료")

        else:
            mqtt_client.publish(MQTT_TOPIC_RESULT, json.dumps({"error": "Not enough data"}))

    except Exception as e:
        print(f"❌ 작업 중 에러 발생: {e}")
        mqtt_client.publish(MQTT_TOPIC_RESULT, json.dumps({"error": str(e)}))

    finally:
        # 작업 종료 처리
        time.sleep(1)  # 쿨다운 (너무 빠른 재요청 방지)
        is_processing = False
        mqtt_client.publish(MQTT_TOPIC_STATUS, "READY")
        print("⏹ [작업 종료] 대기 모드로 전환")


def on_firebase_command(event):
    """파이어베이스 데이터 변경 감지 리스너"""
    global is_processing

    data = event.data
    if not data: return

    print(f"▷ [요청 감지] {data}")

    # 중복 실행 방지 로직
    if is_processing:
        print("   ⛔ [거절] 현재 예측 작업이 진행 중입니다.")
        return

    # 'start' 명령일 때만 실행하도록 조건 추가 가능
    # if data != "start": return

    is_processing = True
    # 쓰레드 시작 (메인 루프가 멈추지 않도록)
    t = threading.Thread(target=prediction_worker, args=(data,))
    t.start()


def main():
    global rtdb_manager, mqtt_client

    print("--- Raspberry Pi AI Bridge 시작 ---")

    # 1. 모델 로드
    load_ai_models()

    # 2. Firebase 연결
    rtdb_manager = RealtimeDatabaseManager(FIREBASE_KEY_PATH, DATABASE_URL)

    # 3. MQTT 연결
    mqtt_client = mqtt.Client(client_id="drying_rack_pi")
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
        mqtt_client.loop_start()  # 별도 쓰레드에서 MQTT 통신 처리
        print(f"✅ MQTT 브로커 연결 성공 ({MQTT_BROKER})")
    except Exception as e:
        print(f"❌ MQTT 연결 실패: {e}")
        return

    # 4. Firebase 리스너 등록
    try:
        ref = db.reference(COMMAND_PATH)
        ref.listen(on_firebase_command)
        print(f"✅ Firebase 리스너 등록 완료 ({COMMAND_PATH})")
        print("🚀 시스템 준비 완료. 명령 대기중...")

        # 메인 쓰레드 유지
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n종료합니다.")
        mqtt_client.loop_stop()


if __name__ == '__main__':
    main()