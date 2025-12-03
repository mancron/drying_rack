import time
import threading
import json
import joblib
import pandas as pd
import numpy as np
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
        bundle = joblib.load('all_sensors_bundle.pkl')

        models = bundle['models']
        scalers = bundle['scalers']
        features_list = bundle['features']

        print(f"✅ 모델 번들 로드 완료 (센서 {list(models.keys())} 모델 포함)")

    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print("   (먼저 drying_time_predictor.py를 실행해 'all_sensors_bundle.pkl'을 생성해주세요)")
        exit()


def get_current_session_data():
    """RTDB에서 전체 데이터를 가져와 '현재 진행 중인 세션'의 데이터만 추출"""
    try:
        df = rtdb_manager.fetch_sequential_paths_as_dataframe(BASE_DATA_PATH)

        if df.empty or len(df) < 5:
            print("⚠ 데이터가 부족하여 예측할 수 없습니다 (최소 5개 필요).")
            return None

        df = df.sort_values(by='timestamp').reset_index(drop=True)
        df['light_lux_avg'] = df['lux1']
        df = df.rename(columns={'temperature': 'ambient_temp', 'humidity': 'ambient_humidity'})

        time_diff = df['timestamp'].diff().dt.total_seconds() / 3600
        df['session_id'] = (time_diff > 2.0).cumsum()

        last_session_id = df['session_id'].max()
        current_session_df = df[df['session_id'] == last_session_id].copy().reset_index(drop=True)

        return current_session_df

    except Exception as e:
        print(f"❌ 데이터 조회/변환 중 오류: {e}")
        return None


def extract_features_for_sensor(session_df, sensor_num):
    """특정 센서에 대한 예측 피처 생성"""
    try:
        if len(session_df) < 5:
            return None, None, None

        sensor_col = f'moisture_percent_{sensor_num}'

        latest_rows = session_df.tail(5).copy()
        latest = latest_rows.iloc[-1]
        prev1 = latest_rows.iloc[-2]

        # [Fix] Numpy 타입을 순수 Python 타입으로 변환
        curr_hum = float(latest[sensor_col])

        delta_hum = curr_hum - float(prev1[sensor_col])
        delta_lux = float(latest['light_lux_avg']) - float(prev1['light_lux_avg'])

        humidity_values = latest_rows[sensor_col].values
        trend = float(np.mean(humidity_values[-3:]))
        variance = float(np.std(humidity_values))

        start_time = session_df['timestamp'].iloc[0]
        time_elapsed = float((latest['timestamp'] - start_time).total_seconds() / 60)

        initial_hum = float(session_df[sensor_col].iloc[0])

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
        }])[features_list]

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
        current_session_df = get_current_session_data()

        if current_session_df is not None:
            predictions = {}
            sensor_results = {}

            for i in range(1, 5):
                if i not in models:
                    continue

                features, curr_hum, delta_hum = extract_features_for_sensor(current_session_df, i)

                if features is not None:
                    scaled_features = scalers[i].transform(features)
                    pred_time = models[i].predict(scaled_features)[0]

                    # [Fix] float()로 감싸서 Numpy 타입 제거
                    pred_time = float(max(0, pred_time))

                    # 상식 기반 보정
                    if curr_hum < 2.0:
                        if delta_hum >= 0:
                            pred_time = 0.0
                        else:
                            pred_time = min(pred_time, 20.0)
                    elif curr_hum < 5.0 and delta_hum >= -0.5:
                        pred_time = min(pred_time, 60.0)

                    predictions[i] = round(pred_time, 1)

                    sensor_results[f"sensor_{i}"] = {
                        "humidity": round(curr_hum, 1),
                        "predicted_min": round(pred_time, 1)
                    }
                    print(f"   ✅ 센서 {i}: 습도 {curr_hum:.1f}% -> {pred_time:.1f}분 예측")

            if predictions:
                # [Fix] 최종 결과도 float 변환
                max_predicted_time = float(max(predictions.values()))

                result_msg = {
                    "predicted_minutes": max_predicted_time,
                    "details": sensor_results,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                # 파이어베이스 저장 (이제 에러 안 남)
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

    load_ai_models()

    rtdb_manager = RealtimeDatabaseManager(FIREBASE_KEY_PATH, DATABASE_URL)

    mqtt_client = mqtt.Client(client_id="drying_rack_pi")
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
        mqtt_client.loop_start()
        print(f"✅ MQTT 브로커 연결 성공 ({MQTT_BROKER})")
    except Exception as e:
        print(f"❌ MQTT 연결 실패: {e}")
        return

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