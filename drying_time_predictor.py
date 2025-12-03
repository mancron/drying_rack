import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
from firebase_manager import RealtimeDatabaseManager
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

"""
[Method 2: 개별 센서 독립 예측 방식]
Firebase DB에서 센서 데이터를 가져와 각 센서(1~4번)를 독립적인 건조 사건으로 분리하여 학습합니다.
예측 시에는 4개 센서의 예상 종료 시간을 각각 구한 뒤, 그 중 가장 늦게 끝나는 시간(Max)을 최종 결과로 반환합니다.
"""


# --------------------------------------------------------------------------
# (1) 데이터 조회 및 전처리
# --------------------------------------------------------------------------

def fetch_all_data_from_rtdb(key_path, db_url, base_data_path):
    """DB에서 전체 데이터를 순차적으로 가져와 병합"""
    try:
        rtdb_manager = RealtimeDatabaseManager(key_path, db_url)
        df = rtdb_manager.fetch_sequential_paths_as_dataframe(base_data_path)
        if df.empty: return pd.DataFrame()
        df.sort_values(by='timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"RTDB 조회 실패: {e}")
        return pd.DataFrame()


def preprocess_data_independent_sensors(df_original,
                                        session_threshold_hours=1.0,
                                        dry_threshold_percent=1.0,
                                        dry_stable_rows=10):
    """
    (핵심 수정) 센서 1~4번을 독립적인 데이터로 쪼개서 학습 데이터를 4배로 늘림.
    각 센서별로 '자기가 마르는 시간'을 정답(Target)으로 설정함.
    """
    if df_original.empty:
        return pd.DataFrame(), pd.Series(), [], pd.Series()

    df = df_original.copy()

    # 1. 컬럼명 표준화
    df['light_lux_avg'] = df['lux1']
    df = df.rename(columns={'temperature': 'ambient_temp', 'humidity': 'ambient_humidity'})
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # 2. 세션 ID 생성
    time_diff = df['timestamp'].diff().dt.total_seconds() / 3600
    df['session_id'] = (time_diff > session_threshold_hours).cumsum()

    print(f"총 {df['session_id'].nunique()}개의 건조 세션 감지. (이제 각 세션을 4개로 쪼갭니다)")

    sensor_columns = [
        'moisture_percent_1', 'moisture_percent_2',
        'moisture_percent_3', 'moisture_percent_4'
    ]

    all_sensor_data = []

    # 3. [이중 루프] 각 세션 안에서 -> 각 센서별로 데이터를 따로 뽑아냄
    for session_id in df['session_id'].unique():
        session_full_df = df[df['session_id'] == session_id].copy()

        for sensor_col in sensor_columns:
            # 해당 센서 데이터만 뽑아서 임시 데이터프레임 생성
            # (주변 환경 정보는 공통으로 가져감)
            sub_df = session_full_df[[
                'timestamp', 'ambient_temp', 'ambient_humidity', 'light_lux_avg',
                sensor_col  # 현재 처리 중인 센서 값만 가져옴
            ]].copy()

            # 컬럼명을 통일 (모델은 어느 센서인지 모르고 'current_humidity'로만 알게 됨)
            sub_df = sub_df.rename(columns={sensor_col: 'current_humidity'})

            # --- 개별 센서의 건조 완료 시점 탐지 ---
            is_dry = sub_df['current_humidity'] < dry_threshold_percent
            is_stable_dry = is_dry.rolling(window=dry_stable_rows).sum() >= dry_stable_rows
            stable_indices = np.where(is_stable_dry)[0]

            if len(stable_indices) > 0:
                # 이 센서가 마른 시점
                dry_idx = stable_indices[0] - dry_stable_rows + 1
                true_end_time = sub_df.iloc[dry_idx]['timestamp']

                # 마른 시점 이후 데이터는 자름 (학습 방해 금지)
                sub_df = sub_df[sub_df['timestamp'] <= true_end_time].copy()

                # y값(남은 시간) 계산
                sub_df['remaining_time_minutes'] = (true_end_time - sub_df['timestamp']).dt.total_seconds() / 60

                # 피처 생성 (변화량, 추세)
                sub_df['delta_humidity'] = sub_df['current_humidity'].diff().fillna(0)
                sub_df['delta_illumination'] = sub_df['light_lux_avg'].diff().fillna(0)
                sub_df['humidity_trend'] = sub_df['current_humidity'].rolling(3).mean().bfill()

                # 그룹 분리를 위한 ID (세션ID 유지)
                sub_df['session_id'] = session_id

                all_sensor_data.append(sub_df)

            else:
                # 이 센서는 끝까지 안 마름 -> 학습에서 제외하거나 전체 시간을 정답으로 씀
                # (여기서는 품질을 위해 제외)
                pass

    if not all_sensor_data:
        print("유효한 학습 데이터를 만들지 못했습니다.")
        return pd.DataFrame(), pd.Series(), [], pd.Series()

    # 4. 데이터 합치기
    processed_df = pd.concat(all_sensor_data, ignore_index=True)

    features = [
        'ambient_temp', 'ambient_humidity', 'light_lux_avg',
        'current_humidity', 'delta_humidity', 'delta_illumination', 'humidity_trend'
    ]
    target = 'remaining_time_minutes'

    processed_df = processed_df.dropna(subset=features + [target])

    X = processed_df[features]
    y = processed_df[target]
    groups = processed_df['session_id']  # 세션 단위 분할을 위해 필요

    print(f"데이터 뻥튀기 완료: 총 {len(processed_df)}개 샘플 생성 (원본 대비 약 4배)")
    return X, y, features, groups


# --------------------------------------------------------------------------
# (2) 모델 학습
# --------------------------------------------------------------------------
# 수정된 create_and_save_model (테스트용)
def create_and_save_model(X, y, groups):
    if X.empty: return None
    print("\n--- [코드 검증용] 전체 데이터 학습 테스트 ---")

    # 1. 데이터를 나누지 않고 통째로 씁니다.
    X_train = X
    y_train = y

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 모델 학습
    model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=500, learning_rate=0.05,
        max_depth=5, random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # 2. 학습한 데이터로 바로 채점 (자가 채점)
    y_pred = model.predict(X_train_scaled)

    r2 = r2_score(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)

    print(f"\n[자가 채점 결과 (Train Score)]")
    print(f"R² Score: {r2:.4f} (이 점수가 0.9 이상 나와야 코드가 정상)")
    print(f"MAE: {mae:.2f}분")

    # (이하 저장 로직은 유지)
    joblib.dump(model, 'drying_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    return model



# --------------------------------------------------------------------------
# (3) 실시간 예측용 함수 (4개 센서 각각 예측 후 MAX)
# --------------------------------------------------------------------------

def make_features_for_independent_prediction(current_session_df, features_list):
    """최신 데이터(3행)를 받아 4개의 피처 세트(센서 1~4용)를 만듭니다."""
    if len(current_session_df) < 3: return None

    df = current_session_df.copy()

    # 공통 컬럼 정리
    df['light_lux_avg'] = df['lux1']
    df = df.rename(columns={'temperature': 'ambient_temp', 'humidity': 'ambient_humidity'})

    latest_rows = df.tail(3)  # 마지막 3개

    input_batch = []  # 모델에 넣을 4개의 행

    sensor_cols = ['moisture_percent_1', 'moisture_percent_2', 'moisture_percent_3', 'moisture_percent_4']

    for sensor in sensor_cols:
        # 각 센서를 'current_humidity'로 간주하고 피처 계산
        latest = latest_rows.iloc[-1]
        prev1 = latest_rows.iloc[-2]
        prev2 = latest_rows.iloc[-3]

        curr_hum = latest[sensor]
        prev_hum = prev1[sensor]
        prev2_hum = prev2[sensor]

        # 피처 계산
        delta_hum = curr_hum - prev_hum
        delta_lux = latest['light_lux_avg'] - prev1['light_lux_avg']
        trend = (curr_hum + prev_hum + prev2_hum) / 3

        row = {
            'ambient_temp': latest['ambient_temp'],
            'ambient_humidity': latest['ambient_humidity'],
            'light_lux_avg': latest['light_lux_avg'],
            'current_humidity': curr_hum,
            'delta_humidity': delta_hum,
            'delta_illumination': delta_lux,
            'humidity_trend': trend
        }
        input_batch.append(row)

    return pd.DataFrame(input_batch)[features_list]  # 순서 맞춰서 반환


# --------------------------------------------------------------------------
# 메인 실행부 (수정됨: 과거 특정 시점으로 돌아가서 예측 테스트)
# --------------------------------------------------------------------------
if __name__ == '__main__':
    KEY_PATH = "firebase.json"
    DB_URL = "https://smart-drying-rack-fe271-default-rtdb.firebaseio.com/"
    BASE_PATH = "drying-rack"

    # 1. 학습 단계
    raw_df = fetch_all_data_from_rtdb(KEY_PATH, DB_URL, BASE_PATH)

    if not raw_df.empty:
        # 전처리 및 모델 학습
        X, y, feats, groups = preprocess_data_independent_sensors(
            raw_df, session_threshold_hours=2.0, dry_threshold_percent=1.0
        )
        model = create_and_save_model(X, y, groups)
    else:
        print("데이터 없음")
        exit()

    # 2. 시뮬레이션 단계 (타임머신 테스트)
    print("\n--- 시뮬레이션 (과거 시점 테스트) ---")
    try:
        loaded_model = joblib.load('drying_model.pkl')
        loaded_scaler = joblib.load('scaler.pkl')

        if not raw_df.empty:
            # (1) 마지막 세션 추출
            df_sim = raw_df.copy().sort_values(by='timestamp')
            time_diff = df_sim['timestamp'].diff().dt.total_seconds() / 3600
            df_sim['session_id'] = (time_diff > 2.0).cumsum()

            last_session_id = df_sim['session_id'].max()
            last_session_df = df_sim[df_sim['session_id'] == last_session_id].copy().reset_index(drop=True)

            # ----------------------------------------------------------------
            # [★ 핵심 수정] 맨 끝(tail)이 아니라, "중간 지점"을 강제로 선택
            # 예: 전체 데이터의 50% 지점 (한창 건조 중일 때)
            # ----------------------------------------------------------------
            test_index = len(last_session_df) // 2  # 딱 중간 지점

            # 만약 특정 습도 시점을 찾고 싶다면 아래 주석 해제:
            # test_index = (last_session_df['moisture_percent_1'] < 30).idxmax() # 습도가 30% 밑으로 떨어지기 직전

            start_time = last_session_df['timestamp'].iloc[0]  # 세션 시작 시간

            # "그 당시"라고 가정하고 데이터 3개만 잘라냄
            current_data_slice = last_session_df.iloc[test_index - 3: test_index]
            current_timestamp = current_data_slice['timestamp'].iloc[-1]

            # (2) 현재 상태 출력
            elapsed_minutes = (current_timestamp - start_time).total_seconds() / 60

            moist_cols = ['moisture_percent_1', 'moisture_percent_2', 'moisture_percent_3', 'moisture_percent_4']
            current_humidity_avg = current_data_slice.iloc[-1][moist_cols].mean()

            print(f"⏱ [타임머신] 현재 시점: 세션 시작 후 {int(elapsed_minutes)}분 경과")
            print(f"💧 현재 평균 습도: {current_humidity_avg:.1f}% (한창 건조 중)")

            # (3) 예측 수행
            batch_inputs = make_features_for_independent_prediction(current_data_slice, feats)

            if batch_inputs is not None:
                scaled_inputs = loaded_scaler.transform(batch_inputs)
                preds = loaded_model.predict(scaled_inputs)

                final_time = max(preds)
                final_time = max(0, final_time)

                print("-" * 30)
                print(f"각 센서별 예측(분): {preds}")
                print(f"✅ AI 예측: 앞으로 {int(final_time)}분 더 돌면 마릅니다.")
                print("-" * 30)

                # (참고) 실제 정답 확인 (미래를 미리 보기)
                real_end_time = last_session_df['timestamp'].max()
                real_remaining = (real_end_time - current_timestamp).total_seconds() / 60
                print(f"👀 (정답지 확인) 실제로는 {int(real_remaining)}분 뒤에 끝났습니다.")
                print(f"🎯 오차: {int(abs(final_time - real_remaining))}분")

    except Exception as e:
        print(f"시뮬레이션 오류: {e}")