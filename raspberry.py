import time
import threading
import json
import joblib
import pandas as pd
import numpy as np  # 수학 연산용 추가
import paho.mqtt.client as mqtt
from firebase_admin import db

# 기존 파일들에서 필요한 클래스 임포트
from firebase_manager import RealtimeDatabaseManager

# --- 설정 (Configuration) ---
FIREBASE_KEY_PATH = "firebase.json"
DATABASE_URL = "https://smart-drying-rack-fe271-default-rtdb.firebaseio.com/"
BASE_DATA_PATH = "drying-rack"
COMMAND_PATH = "/drying-rack/command"

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_RESULT = "drying_rack/prediction_result"
MQTT_TOPIC_STATUS = "drying_rack/status"

# --- 전역 변수 ---
is_processing = False
rtdb_manager = None
mqtt_client = None

# 번들로 저장된 모델 객체들을 담을 변수
models = {}
scalers = {}
features_list = []


def load_ai_models():
    """저장된 통합 AI 모델 번들(all_sensors_bundle.pkl) 불러오기"""
    global models, scalers, features_list
    try:
        # 통합 파일 로드
        bundle = joblib.load('all_sensors_bundle.pkl')

        models = bundle['models']
        scalers = bundle['scalers']
        features_list = bundle['features']

        print(f"✅ 모델 번들 로드 완료 (센서 {list(models.keys())} 모델 포함)")
        print(f"   사용 피처: {features_list}")

    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print("   (먼저 drying_time_predictor.py를 실행해 'all_sensors_bundle.pkl'을 생성해주세요)")
        exit()


def get_current_session_data():
    """
    RTDB에서 전체 데이터를 가져와 '현재 진행 중인 세션'의 데이터만 추출
    (경과 시간 및 초기 습도 계산을 위해 세션 시작점 파악 필수)
    """
    try:
        # 전체 데이터 가져오기 (데이터 양이 많으면 limit 등을 고려해야 함)
        df = rtdb_manager.fetch_sequential_paths_as_dataframe(BASE_DATA_PATH)

        if df.empty or len(df) < 5:
            print("⚠ 데이터가 부족하여 예측할 수 없습니다 (최소 5개 필요).")
            return None

        # 시간순 정렬 및 컬럼 표준화
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        df['light_lux_avg'] = df['lux1']
        df = df.rename(columns={'temperature': 'ambient_temp', 'humidity': 'ambient_humidity'})

        # 세션 분리 로직 (2시간 이상 공백 시 새로운 세션)
        time_diff = df['timestamp'].diff().dt.total_seconds() / 3600
        df['session_id'] = (time_diff > 2.0).cumsum()

        # 가장 마지막(최신) 세션만 추출
        last_session_id = df['session_id'].max()
        current_session_df = df[df['session_id'] == last_session_id].copy().reset_index(drop=True)

        return current_session_df

    except Exception as e:
        print(f"❌ 데이터 조회/변환 중 오류: {e}")
        return None


def extract_features_for_sensor(session_df, sensor_num):
    """특정 센서에 대한 예측 피처 생성 (1행 데이터프레임 반환)"""
    try:
        # 최소 데이터 확인
        if len(session_df) < 5:
            return None

        sensor_col = f'moisture_percent_{sensor_num}'

        # 최근 5개 데이터 (추세/분산 계산용)
        latest_rows = session_df.tail(5).copy()
        latest = latest_rows.iloc[-1]
        prev1 = latest_rows.iloc[-2]

        # 1. 기본 값
        curr_hum = latest[sensor_col]

        # 2. 변화량 (Delta)
        delta_hum = curr_hum - prev1[sensor_col]
        delta_lux = latest['light_lux_avg'] - prev1['light_lux_avg']

        # 3. 추세 및 분산 (Trend & Variance)
        humidity_values = latest_rows[sensor_col].values
        trend = np.mean(humidity_values[-3:])  # 최근 3개 평균
        variance = np.std(humidity_values)  # 최근 5개 표준편차

        # 4. 시간 관련 피처 (Time Elapsed)
        start_time = session_df['timestamp'].iloc[0]  # 세션 시작 시간
        time_elapsed = (latest['timestamp'] - start_time).total_seconds() / 60

        # 5. 초기 값 (Initial Humidity)
        initial_hum = session_df[sensor_col].iloc[0]

        # 피처 데이터프레임 생성
        input_data = pd.DataFrame([{
            'ambient_temp': latest['ambient_temp'],
            'ambient_humidity': latest['ambient_humidity'],
            'light_lux_avg': latest['light_lux_avg'],
            'current_humidity': curr_hum,
            'delta_humidity': delta_hum,
            'delta_illumination': delta_lux,
            'humidity_trend': trend,
            'humidity_variance': variance,
            'time_elapsed': time_elapsed,
            'initial_humidity': initial_hum
        }])[features_list]  # 학습 때와 동일한 컬럼 순서 강제

        return input_data, curr_hum, delta_hum

    except Exception as e:
        print(f"   ⚠️ 센서 {sensor_num} 피처 생성 실패: {e}")
        return None, None, None


def prediction_worker(command_data):
    """별도 쓰레드에서 실행될 예측 작업"""
    global is_processing

    print(f"▶ [작업 시작] 명령: {command_data}")
    mqtt_client.publish(MQTT_TOPIC_STATUS, "BUSY")

    try:
        # 1. 현재 세션 데이터 준비
        current_session_df = get_current_session_data()

        if current_session_df is not None:
            predictions = {}
            sensor_results = {}  # MQTT 상세 전송용

            # 2. 각 센서별 예측 수행 (1~4번)
            for i in range(1, 5):
                if i not in models:
                    continue

                # 피처 추출
                features, curr_hum, delta_hum = extract_features_for_sensor(current_session_df, i)

                if features is not None:
                    # 스케일링 및 예측
                    scaled_features = scalers[i].transform(features)
                    pred_time = models[i].predict(scaled_features)[0]
                    pred_time = max(0, pred_time)  # 음수 방지

                    # ----------------------------------------
                    # 🔧 상식 기반 보정 (Predictor 로직 반영)
                    # ----------------------------------------
                    if curr_hum < 2.0:
                        if delta_hum >= 0:
                            pred_time = 0  # 이미 마름 & 습도 안 떨어짐 -> 완료
                        else:
                            pred_time = min(pred_time, 20)  # 마르는 중이면 최대 20분
                    elif curr_hum < 5.0 and delta_hum >= -0.5:
                        pred_time = min(pred_time, 60)  # 습도 낮은데 변화 적음 -> 최대 60분

                    predictions[i] = round(pred_time, 1)
                    sensor_results[f"sensor_{i}"] = {
                        "humidity": round(curr_hum, 1),
                        "predicted_min": round(pred_time, 1)
                    }
                    print(f"   ✅ 센서 {i}: 습도 {curr_hum:.1f}% -> {pred_time:.1f}분 예측")

            if predictions:
                # 3. 최종 결과 집계 (가장 늦게 마르는 시간 기준)
                max_predicted_time = max(predictions.values())

                # 4. 결과 메시지 구성
                result_msg = {
                    "predicted_minutes": max_predicted_time,  # 대표값 (최대값)
                    "details": sensor_results,  # 센서별 상세 정보
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                # 5. 저장 및 전송
                db.reference("/drying-rack/result").set(result_msg)
                print("✅ 파이어베이스에 결과 저장 완료 (/drying-rack/result)")

                payload = json.dumps(result_msg, ensure_ascii=False)
                mqtt_client.publish(MQTT_TOPIC_RESULT, payload)
                print(f"◀ [예측 성공] 최종 남은 시간: {max_predicted_time:.1f}분 -> MQTT 전송 완료")
            else:
                print("⚠️ 모든 센서에 대한 예측 실패")
                mqtt_client.publish(MQTT_TOPIC_RESULT, json.dumps({"error": "Prediction failed for all sensors"}))

        else:
            mqtt_client.publish(MQTT_TOPIC_RESULT, json.dumps({"error": "Not enough data"}))

    except Exception as e:
        print(f"❌ 작업 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        mqtt_client.publish(MQTT_TOPIC_RESULT, json.dumps({"error": str(e)}))

    finally:
        time.sleep(1)
        is_processing = False
        mqtt_client.publish(MQTT_TOPIC_STATUS, "READY")
        print("⏹ [작업 종료] 대기 모드로 전환")


def on_firebase_command(event):
    """파이어베이스 데이터 변경 감지 리스너"""
    global is_processing

    data = event.data
    if not data: return

    print(f"▷ [요청 감지] {data}")

    if is_processing:
        print("   ⛔ [거절] 현재 예측 작업이 진행 중입니다.")
        return

    is_processing = True
    t = threading.Thread(target=prediction_worker, args=(data,))
    t.start()


def main():
    global rtdb_manager, mqtt_client

    print("--- Raspberry Pi AI Bridge (Multi-Sensor) 시작 ---")

    # 1. 모델 번들 로드
    load_ai_models()

    # 2. Firebase 연결
    rtdb_manager = RealtimeDatabaseManager(FIREBASE_KEY_PATH, DATABASE_URL)

    # 3. MQTT 연결
    mqtt_client = mqtt.Client(client_id="drying_rack_pi")
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
        mqtt_client.loop_start()
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

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n종료합니다.")
        mqtt_client.loop_stop()


if __name__ == '__main__':
    main()