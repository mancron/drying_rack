import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from firebase_manager import RealtimeDatabaseManager


# (★) Realtime Database용 데이터 조회 함수
def fetch_data_from_rtdb(key_path, db_url, data_path):
    """Realtime Database에서 데이터를 가져와 DataFrame으로 변환합니다."""
    try:
        rtdb_manager = RealtimeDatabaseManager(key_path, db_url)
        df = rtdb_manager.fetch_path_as_dataframe(data_path)
        return df
    except Exception as e:
        print(f"RTDB 데이터 조회 실패: {e}")
        return pd.DataFrame()


# (★) "실시간 추세"를 학습하도록 완전히 수정된 전처리 함수
def preprocess_data_for_training(df_original, session_threshold_hours=1):
    """
    (학습용) 원본 데이터를 "실시간 추세" 예측 모델용으로 가공합니다.
    - 시간 간격으로 세션을 분리합니다.
    - 각 세션 내에서 '남은 건조 시간'과 '변화량' 피처를 계산합니다.
    """
    if df_original.empty:
        return pd.DataFrame(), pd.Series()

    df = df_original.copy()

    # --- (★) 수정된 부분 시작 ---

    # 1. 실제 컬럼 이름으로 피처 계산 및 이름 표준화

    # (조도) lux1만 있으므로, 이 값을 light_lux_avg로 사용
    df['light_lux_avg'] = df['lux1']

    # (옷 습도) 4개 센서의 평균을 cloth_humidity로 사용
    moist_cols = ['moisture_percent_1', 'moisture_percent_2', 'moisture_percent_3', 'moisture_percent_4']
    df['cloth_humidity'] = df[moist_cols].mean(axis=1)

    # (주변 온습도) 이름 변경
    df = df.rename(columns={
        # 'ts' 대신 'timestamp'를 사용 (이미 RTDB Manager가 변환해옴)
        'temperature': 'ambient_temp',
        'humidity': 'ambient_humidity'
    })

    # --- (★) 수정된 부분 끝 ---

    # 2. 시간순 정렬
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # 3. 세션 ID 생성 (기존 로직 유지)
    time_diff = df['timestamp'].diff().dt.total_seconds() / 3600
    df['session_id'] = (time_diff > session_threshold_hours).cumsum()

    print(f"총 {df['session_id'].nunique()}개의 건조 세션을 감지했습니다.")

    # 4. 각 세션별로 "실시간 피처" 생성
    all_sessions_data = []
    for session_id in df['session_id'].unique():
        session_df = df[df['session_id'] == session_id].copy()

        # 4-1. (y) 타겟 변수 계산: "남은 건조 시간"
        end_time = session_df['timestamp'].max()
        session_df['remaining_time_minutes'] = (end_time - session_df['timestamp']).dt.total_seconds() / 60

        # 4-2. (X) 실시간 피처 계산: 변화량(delta)과 추세(trend)
        session_df['Δhumidity'] = session_df['cloth_humidity'].diff().fillna(0)
        session_df['Δillumination'] = session_df['light_lux_avg'].diff().fillna(0)
        session_df['humidity_trend'] = session_df['cloth_humidity'].rolling(3).mean().fillna(method='bfill')

        all_sessions_data.append(session_df)

    # 5. 모든 세션 데이터를 다시 하나로 합침
    processed_df = pd.concat(all_sessions_data, ignore_index=True)

    # 6. 학습용 X, y 데이터 분리
    features = [
        'ambient_temp',
        'ambient_humidity',
        'light_lux_avg',
        'cloth_humidity',
        'Δhumidity',
        'Δillumination',
        'humidity_trend'
    ]
    target = 'remaining_time_minutes'

    # 학습에 사용할 수 있는 데이터만 필터링 (NaN 값 등 제외)
    processed_df = processed_df.dropna(subset=features + [target])

    X = processed_df[features]
    y = processed_df[target]

    print("실시간 추세 기반 데이터 전처리 완료.")
    return X, y, features


def create_correlation_heatmap(X, y):
    """상관관계 히트맵 생성 (기존과 동일)"""
    if X.empty:
        print("데이터가 없어 히트맵을 생성할 수 없습니다.")
        return
    print("\n--- 상관관계 히트맵 생성 시작 ---")
    df_for_corr = X.copy()
    df_for_corr['remaining_time_minutes'] = y
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.family'] = 'Malgun Gothic'
    corr = df_for_corr.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('건조 시간과 주요 특징 간의 상관관계 히트맵', fontsize=16, pad=12)
    plt.savefig('correlation_heatmap.png')
    print("상관관계 히트맵을 'correlation_heatmap.png' 파일로 저장했습니다.")
    plt.show()


# (★) 스케일러(StandardScaler)도 함께 저장하도록 수정
def create_and_save_model(X, y):
    """전체 데이터로 모델과 스케일러를 학습하고 파일로 저장합니다."""
    if X.empty or y.empty:
        print("학습할 데이터가 없어 모델 생성을 건너뜁니다.")
        return None

    print("\n--- 모델 및 스케일러 학습 시작 ---")

    # 1. 스케일러 학습 및 적용
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. 모델 학습
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,  # (성능을 위해 estimators 증가)
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_scaled, y)

    # 3. 모델과 스케일러 저장 (★ 스케일러 저장이 필수!)
    joblib.dump(model, 'drying_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("모델('drying_model.pkl')과 스케일러('scaler.pkl')를 저장했습니다.")
    return model


# (★) "실시간 추세" 피처를 생성하도록 수정된 예측 함수
def make_features_for_prediction(current_session_df, features_list):
    """(예측용) 현재 세션 데이터로 실시간 추세 피처를 생성합니다."""
    # 최소 3개 데이터가 필요 (rolling(3) 때문)
    if len(current_session_df) < 3:
        return None

    df = current_session_df.copy()

    # --- (★) 수정된 부분 시작 ---

    # 1. 실제 컬럼 이름으로 피처 계산 및 이름 표준화

    # (조도) lux1만 있으므로, 이 값을 light_lux_avg로 사용
    df['light_lux_avg'] = df['lux1']

    # (옷 습도) 4개 센서의 평균을 cloth_humidity로 사용
    moist_cols = ['moisture_percent_1', 'moisture_percent_2', 'moisture_percent_3', 'moisture_percent_4']
    df['cloth_humidity'] = df[moist_cols].mean(axis=1)

    # (주변 온습도) 이름 변경
    df = df.rename(columns={
        'temperature': 'ambient_temp',
        'humidity': 'ambient_humidity'
    })

    # --- (★) 수정된 부분 끝 ---

    # 2. 가장 마지막 3개 데이터 추출
    latest, prev_1, prev_2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]

    # 3. 실시간 추세 피처 수동 계산
    delta_humidity = latest['cloth_humidity'] - prev_1['cloth_humidity']
    delta_illumination = latest['light_lux_avg'] - prev_1['light_lux_avg']
    humidity_trend = (latest['cloth_humidity'] + prev_1['cloth_humidity'] + prev_2['cloth_humidity']) / 3

    # 4. 모델이 학습한 순서대로 2D 배열 생성
    # (데이터프레임을 잠시 만들어 순서를 보장)
    temp_df = pd.DataFrame([{
        'ambient_temp': latest['ambient_temp'],
        'ambient_humidity': latest['ambient_humidity'],
        'light_lux_avg': latest['light_lux_avg'],
        'cloth_humidity': latest['cloth_humidity'],
        'Δhumidity': delta_humidity,
        'Δillumination': delta_illumination,
        'humidity_trend': humidity_trend
    }])

    # 학습 시 사용한 피처 목록(features_list) 순서대로 정렬
    features = temp_df[features_list].values

    return features


# --- 메인 실행 로직 ---
if __name__ == '__main__':
    # --- 0. 설정 ---
    FIREBASE_KEY_PATH = "firebase.json"
    # (★) Realtime Database URL로 변경
    DATABASE_URL = "https://smart-drying-rack-fe271-default-rtdb.firebaseio.com/"
    # (★) 사용자가 지정한 RTDB 경로로 변경
    DATA_PATH = "drying-rack-readings-1"
    DRYING_COMPLETE_THRESHOLD = 5.0

    # --- 1. 학습 단계 ---
    print("--- RTDB에서 전체 학습 데이터 로드 시작 ---")
    all_completed_data = fetch_data_from_rtdb(FIREBASE_KEY_PATH, DATABASE_URL, DATA_PATH)

    if not all_completed_data.empty:
        all_completed_data.rename(columns={'ts': 'timestamp'}, inplace=True, errors='ignore')
        all_completed_data.sort_values(by='timestamp', inplace=True)

        # (★) 새 전처리 함수 사용
        X, y, trained_features = preprocess_data_for_training(all_completed_data)

        create_correlation_heatmap(X, y)
        create_and_save_model(X, y)
    else:
        print("RTDB에서 데이터를 가져오지 못해 학습을 진행할 수 없습니다.")

    print("\n" + "=" * 50 + "\n")

    # --- 2. 실시간 예측 단계 (시뮬레이션) ---
    print("--- 실시간 예측 시뮬레이션 시작 ---")

    try:
        # (★) 모델과 스케일러 모두 로드
        loaded_model = joblib.load('drying_model.pkl')
        loaded_scaler = joblib.load('scaler.pkl')
        print("저장된 모델과 스케일러를 불러왔습니다.")
    except FileNotFoundError:
        print("저장된 모델 또는 스케일러 파일이 없습니다. 먼저 모델을 학습시켜주세요.")
        exit()

    # 2-2. 시뮬레이션용 데이터 준비
    if not all_completed_data.empty and all_completed_data.shape[0] > 15:
        new_session_data = all_completed_data.tail(15).copy().reset_index(drop=True)

        actual_time_sec = (new_session_data['timestamp'].max() - new_session_data['timestamp'].min()).total_seconds()
        print(f"\n새로운 건조 시작! (시뮬레이션용 데이터 {len(new_session_data)}개)")
        print(f"이 세션의 실제 총 건조 시간: {actual_time_sec / 60:.0f} 분")
        print("-" * 30)

        # (★) 예측 시뮬레이션 (최소 3개 데이터부터 시작)
        for i in range(3, len(new_session_data) + 1):
            current_data_slice = new_session_data.head(i)

            elapsed_minutes = (current_data_slice['timestamp'].max() - current_data_slice[
                'timestamp'].min()).total_seconds() / 60
            latest_humidity = current_data_slice.iloc[-1]['cloth_moist_pct']
            latest_timestamp = current_data_slice.iloc[-1]['timestamp']

            print(
                f"데이터 {i}개 수집 [{latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}] (경과 시간: {elapsed_minutes:.1f}분, 현재 습도: {latest_humidity:.1f}%)")

            # 기능 1: 건조 완료 판단
            if latest_humidity < DRYING_COMPLETE_THRESHOLD:
                print("==> 건조 완료 기준 도달! 건조를 종료합니다.")
                break

            # 기능 2: 남은 시간 예측
            # (★) 새 피처 생성 함수 사용
            prediction_features = make_features_for_prediction(current_data_slice, trained_features)

            if prediction_features is not None:
                # (★) 예측 전 스케일링 필수!
                scaled_features = loaded_scaler.transform(prediction_features)

                # (★) 모델이 '남은 시간'을 바로 예측함
                predicted_remaining_time = loaded_model.predict(scaled_features)[0]
                predicted_remaining_time = max(0, predicted_remaining_time)  # 0분 이하 방지

                print(f"==> 예상 남은 시간: {predicted_remaining_time:.2f} 분")
    else:
        print("시뮬레이션을 위한 데이터가 부족합니다.")