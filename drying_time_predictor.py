import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
# (★) matplotlib, seaborn 임포트 제거
from sklearn.preprocessing import StandardScaler
from firebase_manager import RealtimeDatabaseManager
from heatmap_generator import create_correlation_heatmap  # (★) 새 파일에서 임포트
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
"""
Firebase DB에서 원본(Raw) 센서 데이터를 가져와 휴지기를 제거하고 건조 세션별로 분리
각 세션의 데이터(현재값, 변화량)와 남은 건조 시간(정답)을 계산하여 AI 모델을 학습시키고 파일(.pkl)로 저장
저장된 모델을 불러와, 새로 들어오는 데이터에 대한 예상 남은 시간을 실시간으로 예측(시뮬레이션)
"""


# (★) Realtime Database용 데이터 조회 함수 (순차 조회 로직)
def fetch_all_data_from_rtdb(key_path, db_url, base_data_path):
    """
    Realtime Database에서 base_data_path-1, -2, ... 경로의 데이터를 순차적으로 가져와
    하나의 DataFrame으로 병합하고 정렬합니다.
    """
    try:
        # 1. RealtimeDatabaseManager 객체 생성 (URL 전달 필수)
        rtdb_manager = RealtimeDatabaseManager(key_path, db_url)
        # 2. (firebase_manager.py에 추가된 함수) 순차적 데이터 조회
        df = rtdb_manager.fetch_sequential_paths_as_dataframe(base_data_path)

        if df.empty:
            return pd.DataFrame()

        # 시간순 정렬 보장
        df.sort_values(by='timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"RTDB 전체 데이터 조회 실패: {e}")
        return pd.DataFrame()


# (★) "실시간 추세" 및 "진짜 건조 완료" 로직이 적용된 전처리 함수
# (★) "실시간 추세" 및 "진짜 건조 완료" 로직이 적용된 전처리 함수
def preprocess_data_for_training(df_original,
                                 session_threshold_hours=1,
                                 dry_threshold_percent=1.0,  # (★) 추가
                                 dry_stable_rows=10):  # (★) 추가
    """
    (학습용) 원본 데이터를 "실시간 추세" 예측 모델용으로 가공합니다.
    - 시간 간격으로 세션을 분리합니다.
    - (★) 낮은 습도가 지속되는 시점을 '진짜' 종료 시점으로 간주하고 데이터를 자릅니다.
    - (★수정) 건조가 완료되지 않은(중단된) 세션은 학습에서 제외합니다.
    - 각 세션 내에서 '남은 건조 시간'과 '변화량' 피처를 계산합니다.
    """
    if df_original.empty:
        return pd.DataFrame(), pd.Series(), []

    df = df_original.copy()

    # 1. 실제 컬럼 이름으로 피처 계산 및 이름 표준화
    df['light_lux_avg'] = df['lux1']
    moist_cols = ['moisture_percent_1', 'moisture_percent_2', 'moisture_percent_3', 'moisture_percent_4']
    df['cloth_humidity'] = df[moist_cols].mean(axis=1)
    df = df.rename(columns={
        'temperature': 'ambient_temp',
        'humidity': 'ambient_humidity'
    })

    # 2. 시간순 정렬
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # 3. 세션 ID 생성
    time_diff = df['timestamp'].diff().dt.total_seconds() / 3600
    df['session_id'] = (time_diff > session_threshold_hours).cumsum()

    print(f"총 {df['session_id'].nunique()}개의 건조 세션을 감지했습니다.")
    print(f" (★) 건조 완료 기준: 습도 < {dry_threshold_percent}%가 {dry_stable_rows}개 포인트 연속 유지 시")

    # 4. 각 세션별로 "실시간 피처" 생성
    all_sessions_data = []
    for session_id in df['session_id'].unique():
        session_df = df[df['session_id'] == session_id].copy()

        # (★) --- "진짜" 건조 완료 시점 탐지 ---
        is_dry = session_df['cloth_humidity'] < dry_threshold_percent
        is_stable_dry = is_dry.rolling(window=dry_stable_rows).sum() >= dry_stable_rows
        stable_indices_loc = np.where(is_stable_dry)[0]  # .iloc 위치

        if len(stable_indices_loc) > 0:
            first_stable_end_iloc = stable_indices_loc[0]
            first_stable_start_iloc = first_stable_end_iloc - dry_stable_rows + 1
            true_end_timestamp = session_df.iloc[first_stable_start_iloc]['timestamp']

            print(f"  (세션 {session_id}) '진짜' 건조 완료 시점 감지: {true_end_timestamp}")

            # "완료 이후의 데이터(유휴 데이터)를 자름"
            session_df = session_df[session_df['timestamp'] <= true_end_timestamp].copy()

        else:
            # (★ 중요 수정) 건조 완료 조건에 도달하지 못한 세션은 학습에서 제외
            print(f"  (세션 {session_id}) 안정된 건조 상태를 감지하지 못함. >> 학습 데이터에서 제외합니다. <<")
            continue  # 이 세션은 all_sessions_data에 추가하지 않고 건너뜀

        # 4-1. (y) 타겟 변수 계산: "남은 건조 시간"
        end_time = session_df['timestamp'].max()
        session_df['remaining_time_minutes'] = (end_time - session_df['timestamp']).dt.total_seconds() / 60

        # 4-2. (X) 실시간 피처 계산: 변화량(delta)과 추세(trend)
        session_df['Δhumidity'] = session_df['cloth_humidity'].diff().fillna(0)
        session_df['Δillumination'] = session_df['light_lux_avg'].diff().fillna(0)
        session_df['humidity_trend'] = session_df['cloth_humidity'].rolling(3).mean().bfill()

        all_sessions_data.append(session_df)

    # (예외 처리) 유효한 세션이 하나도 없을 경우
    if not all_sessions_data:
        print("경고: 건조 완료 기준을 만족하는 세션이 하나도 없습니다. 임계값(DRY_THRESHOLD)을 확인하세요.")
        return pd.DataFrame(), pd.Series(), []

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


    if processed_df.empty:
        print("전처리 후 남은 데이터가 없습니다.")
        return pd.DataFrame(), pd.Series(), []
    groups = processed_df['session_id']
    X = processed_df[features]
    y = processed_df[target]

    print("실시간 추세 기반 데이터 전처리 완료.")
    return X, y, features , groups

'''
def plot_prediction_results(y_true, y_pred):
    plt.figure(figsize=(10, 6))

    # 산점도 그리기
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue', label='Data Points')

    # 정답 라인 (y = x)
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

    plt.title('Actual vs Predicted Remaining Time')
    plt.xlabel('Actual Remaining Time (min)')
    plt.ylabel('Predicted Remaining Time (min)')
    plt.legend()
    plt.grid(True)
    plt.show()
'''
# (★) 스케일러(StandardScaler)도 함께 저장하도록 수정
def create_and_save_model(X, y,groups):
    """모델 학습 및 성능 평가"""
    if X.empty or y.empty:
        print("학습할 데이터가 없어 모델 생성을 건너뜁니다.")
        return None

    print("\n--- 모델 성능 평가 및 학습 시작 ---")

    # 1. 데이터 분리 (랜덤 분할 -> 그룹 분할로 변경)
    # 세션 ID(groups)를 기준으로 훈련/테스트 셋을 나눔
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print(f"학습 데이터 개수: {len(X_train)} (세션 {groups.iloc[train_idx].nunique()}개)")
    print(f"테스트 데이터 개수: {len(X_test)} (세션 {groups.iloc[test_idx].nunique()}개)")

    # 2. 스케일러 학습
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. 모델 학습 (기존과 동일)
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    # 4. (★) 성능 평가
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)

    # [수정] 옵션 대신 직접 계산 (호환성 해결)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)
    print(f"\n[모델 평가 결과]")
    print(f"1. 평균 오차 (MAE): {mae:.2f}분 (평균적으로 {mae:.2f}분 정도 틀림)")
    print(f"2. RMSE: {rmse:.2f}분")
    print(f"3. 정확도 (R² Score): {r2:.4f} (1.0에 가까울수록 좋음)")
    #plot_prediction_results(y_test, y_pred)
    # -------------------------------------------------------
    # 5. (선택) 실제 서비스용 모델은 전체 데이터로 다시 재학습해서 저장
    print("\n[최종 모델 저장]")
    full_scaler = StandardScaler()
    X_scaled_full = full_scaler.fit_transform(X)  # 전체 데이터 사용

    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    final_model.fit(X_scaled_full, y)

    joblib.dump(final_model, 'drying_model.pkl')
    joblib.dump(full_scaler, 'scaler.pkl')
    print("전체 데이터로 재학습된 모델과 스케일러를 저장했습니다.")

    return final_model


# (★) "실시간 추세" 피처를 생성하도록 수정된 예측 함수
def make_features_for_prediction(current_session_df, features_list):
    """(예측용) 현재 세션 데이터로 실시간 추세 피처를 생성합니다."""
    # 최소 3개 데이터가 필요 (rolling(3) 때문)
    if len(current_session_df) < 3:
        return None

    df = current_session_df.copy()

    # --- (★) 수정된 부분 시작 ---

    # 1. 실제 컬럼 이름으로 피처 계산 및 이름 표준화
    df['light_lux_avg'] = df['lux1']
    moist_cols = ['moisture_percent_1', 'moisture_percent_2', 'moisture_percent_3', 'moisture_percent_4']
    df['cloth_humidity'] = df[moist_cols].mean(axis=1)
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
    features = temp_df[features_list]

    return features


# --- 메인 실행 로직 ---
if __name__ == '__main__':
    # --- 0. 설정 ---
    FIREBASE_KEY_PATH = "firebase.json"
    # (★) Realtime Database URL로 변경
    DATABASE_URL = "https://smart-drying-rack-fe271-default-rtdb.firebaseio.com/"
    # (★) 사용자가 지정한 RTDB 경로의 기본 이름으로 변경
    BASE_DATA_PATH = "drying-rack"
    DRYING_COMPLETE_THRESHOLD = 1.0  # (★) 시뮬레이션 종료 시점 습도

    # (★) --- 새 파라미터 (viz.py와 동일) --- (★)
    SESSION_THRESHOLD_HOURS = 2.0  # 세션 분리 기준 시간 (1시간)
    DRY_THRESHOLD = 1.0  # 학습시 '건조 완료'로 간주할 습도 (1%)
    DRY_STABLE_POINTS = 10  # 위 습도가 연속으로 유지되어야 하는 데이터 개수 (10개)
    # (★) --- --- (★)

    # --- 1. 학습 단계 ---
    print("--- RTDB에서 전체 학습 데이터 로드 시작 ---")
    # (★) 순차 조회 함수 호출로 변경
    all_completed_data = fetch_all_data_from_rtdb(FIREBASE_KEY_PATH, DATABASE_URL, BASE_DATA_PATH)

    # 메인 실행부 수정
    if not all_completed_data.empty:
        # 1. 전처리 함수 호출 (groups 받아오기)
        X, y, trained_features, groups = preprocess_data_for_training(
            all_completed_data.copy(),
            session_threshold_hours=SESSION_THRESHOLD_HOURS,
            dry_threshold_percent=DRY_THRESHOLD,
            dry_stable_rows=DRY_STABLE_POINTS
        )

        # 2. 모델 생성 함수 호출 (groups 넘겨주기)
        create_and_save_model(X, y, groups)
        #create_correlation_heatmap(X, y)
    else:
        print("RTDB에서 데이터를 가져오지 못해 학습을 진행할 수 없습니다.")

    print("\n" + "=" * 50 + "\n")

    # --- 2. 실시간 예측 단계 (시뮬레이션) ---
    print("--- 실시간 예측 시뮬레이션 시작 ---")

    try:
        loaded_model = joblib.load('drying_model.pkl')
        loaded_scaler = joblib.load('scaler.pkl')
        print("저장된 모델과 스케일러를 불러왔습니다.")
    except FileNotFoundError:
        print("저장된 모델 또는 스케일러 파일이 없습니다. 먼저 모델을 학습시켜주세요.")
        exit()

    # 2-2. 시뮬레이션용 데이터 준비
    real_session_start_time = None  # (★ 수정) 전체 세션 시작 시간을 저장할 변수

    if not all_completed_data.empty and all_completed_data.shape[0] > 10:
        time_diff_sim = all_completed_data['timestamp'].diff().dt.total_seconds() / 3600
        all_completed_data['session_id'] = (time_diff_sim > SESSION_THRESHOLD_HOURS).cumsum()

        last_session_id = all_completed_data['session_id'].max()
        sim_data_pool = all_completed_data[all_completed_data['session_id'] == last_session_id]

        # (★ 수정) 전체 세션의 진짜 시작 시간을 미리 추출
        if not sim_data_pool.empty:
            real_session_start_time = sim_data_pool['timestamp'].min()

        if sim_data_pool.shape[0] > 10:
            new_session_data = sim_data_pool.tail(10).copy().reset_index(drop=True)
            print(f"시뮬레이션용 데이터 {len(new_session_data)}개 준비 완료 (마지막 세션 {last_session_id}의 끝 10개 데이터)")
            print(f"(★) 세션 시작 시간: {real_session_start_time}")  # 확인용 출력
        else:
            print("시뮬레이션을 위한 데이터가 부족합니다. (마지막 세션 데이터 10개 미만)")
            new_session_data = pd.DataFrame()
    else:
        print("시뮬레이션을 위한 데이터가 부족합니다.")
        new_session_data = pd.DataFrame()

    # (★) new_session_data가 비어있지 않을 때만 시뮬레이션 실행
    if not new_session_data.empty and real_session_start_time is not None:
        print(f"\n새로운 건조 시작! (시뮬레이션용 데이터 {len(new_session_data)}개)")
        print("-" * 30)

        final_pred_min = 0
        final_elapsed_min = 0

        for i in range(3, len(new_session_data) + 1):
            current_data_slice = new_session_data.head(i)
            latest_row = current_data_slice.iloc[-1]
            latest_timestamp = latest_row['timestamp']

            # (★ 수정) 경과 시간 = 현재 데이터 시간 - 전체 세션의 진짜 시작 시간
            elapsed_minutes = (latest_timestamp - real_session_start_time).total_seconds() / 60
            final_elapsed_min = elapsed_minutes

            moist_cols = ['moisture_percent_1', 'moisture_percent_2', 'moisture_percent_3', 'moisture_percent_4']
            latest_humidity = latest_row[moist_cols].mean()

            # (기존 출력 유지)
            print(
                f"데이터 {i}개 수집 [{latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}] (경과 시간: {elapsed_minutes:.1f}분, 현재 습도: {latest_humidity:.1f}%)")

            # 기능 1: 건조 완료 판단
            if latest_humidity < DRYING_COMPLETE_THRESHOLD:
                print("==> 건조 완료 기준 도달! 건조를 종료합니다.")
                final_pred_min = 0
                break

            # 기능 2: 남은 시간 예측
            prediction_features = make_features_for_prediction(current_data_slice, trained_features)

            if prediction_features is not None:
                scaled_features = loaded_scaler.transform(prediction_features)
                predicted_remaining_time = loaded_model.predict(scaled_features)[0]
                predicted_remaining_time = max(0, predicted_remaining_time)

                final_pred_min = predicted_remaining_time

                print(f"==> 예상 남은 시간: {int(predicted_remaining_time)}분")

        # (★) 요청하신 포맷 출력
        print(f"예측시간:{int(final_pred_min)}(min)  경과시간:{int(final_elapsed_min)}(min)")

    else:
        print("시뮬레이션을 실행하지 않았습니다.")