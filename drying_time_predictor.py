import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
from firebase_manager import RealtimeDatabaseManager
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

"""
[Option 1: 센서별 독립 모델]
- 센서 1~4번 각각 별도 모델 학습
- 얇은 옷 / 두꺼운 옷 패턴을 개별 학습
- 예측 시 각 센서에 맞는 모델 사용
"""


# --------------------------------------------------------------------------
# (1) 데이터 조회
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


# --------------------------------------------------------------------------
# (2) 센서별 데이터 분리 전처리
# --------------------------------------------------------------------------
def preprocess_data_per_sensor(df_original, sensor_num,
                               session_threshold_hours=2.0,
                               dry_threshold_percent=1.0,
                               dry_stable_rows=10):
    """
    특정 센서 번호(1~4)의 데이터만 추출해서 전처리
    """
    if df_original.empty:
        return pd.DataFrame(), pd.Series(), [], pd.Series()

    df = df_original.copy()
    df['light_lux_avg'] = df['lux1']
    df = df.rename(columns={'temperature': 'ambient_temp', 'humidity': 'ambient_humidity'})
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # 세션 ID 생성
    time_diff = df['timestamp'].diff().dt.total_seconds() / 3600
    df['session_id'] = (time_diff > session_threshold_hours).cumsum()

    sensor_col = f'moisture_percent_{sensor_num}'
    all_sensor_data = []

    # 각 세션별로 처리
    for session_id in df['session_id'].unique():
        session_df = df[df['session_id'] == session_id].copy()

        sub_df = session_df[[
            'timestamp', 'ambient_temp', 'ambient_humidity', 'light_lux_avg', sensor_col
        ]].copy()

        sub_df = sub_df.rename(columns={sensor_col: 'current_humidity'})

        # 건조 완료 시점 탐지
        is_dry = sub_df['current_humidity'] < dry_threshold_percent
        is_stable_dry = is_dry.rolling(window=dry_stable_rows).sum() >= dry_stable_rows
        stable_indices = np.where(is_stable_dry)[0]

        if len(stable_indices) > 0:
            dry_idx = stable_indices[0] - dry_stable_rows + 1
            true_end_time = sub_df.iloc[dry_idx]['timestamp']
            sub_df = sub_df[sub_df['timestamp'] <= true_end_time].copy()

            # 남은 시간 계산
            sub_df['remaining_time_minutes'] = (true_end_time - sub_df['timestamp']).dt.total_seconds() / 60

            # 피처 생성
            sub_df['delta_humidity'] = sub_df['current_humidity'].diff().fillna(0)
            sub_df['delta_illumination'] = sub_df['light_lux_avg'].diff().fillna(0)
            sub_df['humidity_trend'] = sub_df['current_humidity'].rolling(3).mean().bfill()
            sub_df['humidity_variance'] = sub_df['current_humidity'].rolling(5).std().fillna(0)

            # 경과 시간
            start_time = sub_df['timestamp'].iloc[0]
            sub_df['time_elapsed'] = (sub_df['timestamp'] - start_time).dt.total_seconds() / 60

            # 초기 습도 (센서별 특성)
            sub_df['initial_humidity'] = sub_df['current_humidity'].iloc[0]

            sub_df['session_id'] = session_id
            all_sensor_data.append(sub_df)

    if not all_sensor_data:
        return pd.DataFrame(), pd.Series(), [], pd.Series()

    processed_df = pd.concat(all_sensor_data, ignore_index=True)

    features = [
        'ambient_temp', 'ambient_humidity', 'light_lux_avg',
        'current_humidity', 'delta_humidity', 'delta_illumination',
        'humidity_trend', 'humidity_variance', 'time_elapsed', 'initial_humidity'
    ]
    target = 'remaining_time_minutes'

    processed_df = processed_df.dropna(subset=features + [target])

    X = processed_df[features]
    y = processed_df[target]
    groups = processed_df['session_id']

    return X, y, features, groups


# --------------------------------------------------------------------------
# (3) 센서별 모델 학습
# --------------------------------------------------------------------------
def train_sensor_model(X, y, groups, sensor_num):
    """특정 센서용 모델 학습"""
    if X.empty:
        print(f"   ❌ 센서 {sensor_num}: 데이터 없음")
        return None, None

    print(f"\n   🔧 센서 {sensor_num} 모델 학습 중...")
    print(f"      샘플 수: {len(X)}개")

    # Train/Val 분리
    unique_sessions = groups.unique()
    n_sessions = len(unique_sessions)

    print(f"      세션 수: {n_sessions}개")

    if n_sessions < 3:
        print(f"      ⚠️  세션 부족 → 전체 데이터로 학습")
        use_validation = False
        X_train = X
        y_train = y
    else:
        use_validation = True
        np.random.seed(42)
        shuffled_sessions = np.random.permutation(unique_sessions)
        split_point = int(len(shuffled_sessions) * 0.8)

        train_sessions = shuffled_sessions[:split_point]
        val_sessions = shuffled_sessions[split_point:]

        train_mask = groups.isin(train_sessions)
        val_mask = groups.isin(val_sessions)

        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if use_validation:
        X_val_scaled = scaler.transform(X_val)

    # 모델 학습
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        gamma=3.0,
        min_child_weight=10,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42
    )

    if use_validation:
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )

        # 성능 평가
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)

        print(f"      📊 Train MAE: {train_mae:.1f}분 | Val MAE: {val_mae:.1f}분")

        if val_mae - train_mae > 80:
            print(f"      ⚠️  과적합 의심 (차이 {val_mae - train_mae:.1f}분)")
        else:
            print(f"      ✅ 괜찮음 (차이 {val_mae - train_mae:.1f}분)")
    else:
        model.fit(X_train_scaled, y_train, verbose=False)
        train_mae = mean_absolute_error(y_train, model.predict(X_train_scaled))
        print(f"      📊 Train MAE: {train_mae:.1f}분")

    return model, scaler


# --------------------------------------------------------------------------
# (4) 전체 센서 모델 생성 및 저장
# --------------------------------------------------------------------------
def create_all_sensor_models(raw_df):
    """센서 1~4번 모델을 각각 학습"""
    print("\n" + "=" * 60)
    print("🚀 센서별 독립 모델 학습 시작")
    print("=" * 60)

    models = {}
    scalers = {}
    features_list = None

    for sensor_num in range(1, 5):
        print(f"\n📍 센서 {sensor_num} 처리 중...")

        X, y, feats, groups = preprocess_data_per_sensor(
            raw_df, sensor_num,
            session_threshold_hours=2.0,
            dry_threshold_percent=1.0
        )

        if not X.empty:
            model, scaler = train_sensor_model(X, y, groups, sensor_num)

            if model is not None:
                models[sensor_num] = model
                scalers[sensor_num] = scaler
                features_list = feats

                # 개별 저장
                joblib.dump(model, f'sensor_{sensor_num}_model.pkl')
                joblib.dump(scaler, f'sensor_{sensor_num}_scaler.pkl')
        else:
            print(f"   ❌ 센서 {sensor_num}: 전처리 실패")

    # 통합 저장
    if models:
        joblib.dump({
            'models': models,
            'scalers': scalers,
            'features': features_list
        }, 'all_sensors_bundle.pkl')

        print("\n" + "=" * 60)
        print(f"💾 총 {len(models)}개 센서 모델 저장 완료")
        print("   - sensor_1_model.pkl ~ sensor_4_model.pkl")
        print("   - all_sensors_bundle.pkl (통합 파일)")
        print("=" * 60)

    return models, scalers, features_list


# --------------------------------------------------------------------------
# (5) 실시간 예측용 함수 (센서별 모델 사용)
# --------------------------------------------------------------------------
def predict_with_sensor_models(current_session_df, models, scalers, features_list):
    """센서별 모델을 사용해 각각 예측"""
    if len(current_session_df) < 5:
        print("⚠️  예측에 최소 5개의 데이터 포인트가 필요합니다.")
        return None

    df = current_session_df.copy()
    df['light_lux_avg'] = df['lux1']
    df = df.rename(columns={'temperature': 'ambient_temp', 'humidity': 'ambient_humidity'})

    latest_rows = df.tail(5)
    predictions = {}

    sensor_cols = {
        1: 'moisture_percent_1',
        2: 'moisture_percent_2',
        3: 'moisture_percent_3',
        4: 'moisture_percent_4'
    }

    for sensor_num, sensor_col in sensor_cols.items():
        if sensor_num not in models:
            print(f"   ⚠️  센서 {sensor_num} 모델 없음")
            continue

        # 피처 생성
        latest = latest_rows.iloc[-1]
        prev1 = latest_rows.iloc[-2]

        curr_hum = latest[sensor_col]
        prev_hum = prev1[sensor_col]

        delta_hum = curr_hum - prev_hum
        delta_lux = latest['light_lux_avg'] - prev1['light_lux_avg']

        humidity_values = latest_rows[sensor_col].values
        trend = np.mean(humidity_values[-3:])
        variance = np.std(humidity_values)

        start_time = latest_rows['timestamp'].iloc[0]
        time_elapsed = (latest['timestamp'] - start_time).total_seconds() / 60
        initial_hum = latest_rows[sensor_col].iloc[0]

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

        # 예측
        scaled = scalers[sensor_num].transform(input_data)
        pred = models[sensor_num].predict(scaled)[0]
        pred = max(0, pred)  # 음수 방지

        # 🔧 상식 기반 보정 (핵심!)
        if curr_hum < 2.0:  # 거의 마름
            if delta_hum >= 0:  # 더 마르지 않음
                pred = 0  # 즉시 완료
                print(f"   ✅ 센서 {sensor_num}: 이미 건조 완료 (습도 {curr_hum:.1f}%)")
            else:  # 약간 마르는 중
                pred = min(pred, 20)  # 최대 20분
                print(f"   🔧 센서 {sensor_num}: 거의 완료, 예측 조정 → {int(pred)}분")

        elif curr_hum < 5.0 and delta_hum >= -0.5:  # 5% 이하인데 변화 없음
            pred = min(pred, 60)  # 최대 1시간
            print(f"   🔧 센서 {sensor_num}: 습도 낮음, 예측 조정 → {int(pred)}분")

        predictions[sensor_num] = pred

    return predictions


# --------------------------------------------------------------------------
# 메인 실행부
# --------------------------------------------------------------------------
if __name__ == '__main__':
    KEY_PATH = "firebase.json"
    DB_URL = "https://smart-drying-rack-fe271-default-rtdb.firebaseio.com/"
    BASE_PATH = "drying-rack"

    print("\n🚀 센서별 독립 모델 학습 시스템\n")

    # 1. 데이터 로드
    raw_df = fetch_all_data_from_rtdb(KEY_PATH, DB_URL, BASE_PATH)

    if raw_df.empty:
        print("❌ 데이터가 없습니다.")
        exit()

    # 2. 센서별 모델 학습
    models, scalers, features = create_all_sensor_models(raw_df)

    if not models:
        print("❌ 모델 생성 실패")
        exit()

    # 3. 타임머신 테스트
    print("\n" + "=" * 60)
    print("⏰ 타임머신 시뮬레이션 (센서별 독립 모델)")
    print("=" * 60)

    try:
        # 모델 로드
        bundle = joblib.load('all_sensors_bundle.pkl')
        loaded_models = bundle['models']
        loaded_scalers = bundle['scalers']
        loaded_features = bundle['features']

        # 마지막 세션 추출
        df_sim = raw_df.copy().sort_values(by='timestamp')
        time_diff = df_sim['timestamp'].diff().dt.total_seconds() / 3600
        df_sim['session_id'] = (time_diff > 2.0).cumsum()

        last_session_id = df_sim['session_id'].max()
        last_session_df = df_sim[df_sim['session_id'] == last_session_id].copy().reset_index(drop=True)

        if len(last_session_df) > 10:
            test_index = len(last_session_df) // 2
            start_time = last_session_df['timestamp'].iloc[0]
            current_data_slice = last_session_df.iloc[max(0, test_index - 5): test_index]
            current_timestamp = current_data_slice['timestamp'].iloc[-1]

            elapsed_minutes = (current_timestamp - start_time).total_seconds() / 60

            moist_cols = ['moisture_percent_1', 'moisture_percent_2',
                          'moisture_percent_3', 'moisture_percent_4']
            current_humidities = current_data_slice.iloc[-1][moist_cols]

            print(f"\n⏱  현재 시점: {int(elapsed_minutes)}분 경과")
            print(f"💧 센서별 현재 습도:")
            for i, col in enumerate(moist_cols, 1):
                print(f"   센서 {i}: {current_humidities[col]:.1f}%")

            # 센서별 예측
            predictions = predict_with_sensor_models(
                current_data_slice, loaded_models, loaded_scalers, loaded_features
            )

            if predictions:
                print(f"\n🤖 센서별 AI 예측:")
                for sensor_num, pred in predictions.items():
                    print(f"   센서 {sensor_num}: {int(pred)}분")

                # 실제 정답 (센서별)
                real_end_time = last_session_df['timestamp'].max()
                real_remaining = (real_end_time - current_timestamp).total_seconds() / 60

                print(f"\n📊 성능 분석:")
                pred_values = list(predictions.values())
                final_pred = max(pred_values)

                print(f"   최종 예측 (MAX): {int(final_pred)}분")
                print(f"   실제 정답: {int(real_remaining)}분")
                print(f"   오차: {int(abs(final_pred - real_remaining))}분")
                print(f"   정확도: {100 - abs(final_pred - real_remaining) / real_remaining * 100:.1f}%")

                # 센서별 편차 분석
                pred_std = np.std(pred_values)
                print(f"\n   예측 표준편차: {pred_std:.1f}분")

                if pred_std > 100:
                    print(f"   ⚠️  센서 간 편차 큼 → 옷감 특성 차이 반영됨")
                else:
                    print(f"   ✅ 센서 간 편차 작음 → 균일한 건조")

                print("=" * 60)

        else:
            print("테스트 데이터가 부족합니다.")

    except Exception as e:
        print(f"❌ 시뮬레이션 오류: {e}")
        import traceback

        traceback.print_exc()