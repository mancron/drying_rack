import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from firebase_manager import FirestoreManager  # FirestoreManager 클래스 임포트


def fetch_data_from_firestore(key_path, collection_path):
    """Firestore에서 데이터를 가져와 DataFrame으로 변환합니다."""
    try:
        fs_manager = FirestoreManager(key_path)
        df = fs_manager.fetch_collection_as_dataframe(collection_path)
        return df
    except Exception as e:
        print(f"Firestore 데이터 조회 실패: {e}")
        return pd.DataFrame()


def preprocess_data_for_training(df_original, session_threshold_hours=1):
    """
    (학습용) Firestore에서 가져온 원본 데이터를 모델 학습용으로 가공합니다.
    시간 간격을 기준으로 자동으로 건조 세션을 구분합니다.
    """
    if df_original.empty:
        return pd.DataFrame(), pd.Series()

    # 원본 데이터프레임을 수정하지 않도록 복사본을 만듭니다.
    df = df_original.copy()

    # 1. 조도 센서 값들의 평균을 계산하여 새로운 컬럼 생성
    light_cols = ['light_lux_0', 'light_lux_1', 'light_lux_2', 'light_lux_3']
    df['light_lux_avg'] = df[light_cols].mean(axis=1)

    # 2. 모델이 사용할 컬럼 이름으로 변경
    df = df.rename(columns={
        'ts': 'timestamp',
        'temp_c': 'ambient_temp',
        'hum_pct': 'ambient_humidity',
        'cloth_moist_pct': 'cloth_humidity_avg'
    })

    # 3. 시간순으로 정렬
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # 4. 시간 차이를 계산하여 세션 ID 생성
    time_diff = df['timestamp'].diff().dt.total_seconds() / 3600
    # 지정된 시간(예: 1시간) 이상 차이가 나면 새로운 세션으로 간주
    df['session_id'] = (time_diff > session_threshold_hours).cumsum()

    # 5. 세션별로 특징 집계
    agg_features = df.groupby('session_id').agg(
        avg_temp=('ambient_temp', 'mean'),
        avg_humidity=('ambient_humidity', 'mean'),
        avg_light=('light_lux_avg', 'mean'),
        drying_time_minutes=('timestamp', lambda x: (x.max() - x.min()).total_seconds() / 60)
    )
    # 각 세션의 첫 번째 데이터(초기 상태)를 가져옴
    initial_conditions = df.loc[df.groupby('session_id')['timestamp'].idxmin()].set_index('session_id')

    # 집계된 특징과 초기 상태를 결합
    processed_df = agg_features.join(initial_conditions[['cloth_humidity_avg']])

    # 6. 학습용 X, y 데이터 분리
    # 누락된 값이 있는 행은 학습에서 제외
    processed_df.dropna(inplace=True)
    X = processed_df[['avg_temp', 'avg_humidity', 'avg_light', 'cloth_humidity_avg']]
    y = processed_df['drying_time_minutes']

    print("데이터 전처리 및 세션 구분이 완료되었습니다.")
    return X, y


def create_correlation_heatmap(X, y):
    """
    주요 특징들과 건조 시간 간의 상관관계를 보여주는 히트맵을 생성하고 저장합니다.

    Args:
        X (pd.DataFrame): 모델 학습에 사용될 특징 데이터프레임
        y (pd.Series): 목표 변수 (건조 시간)
    """
    if X.empty:
        print("데이터가 없어 히트맵을 생성할 수 없습니다.")
        return

    print("\n--- 상관관계 히트맵 생성 시작 ---")

    # 분석을 위해 특징(X)과 목표 변수(y)를 하나의 데이터프레임으로 합칩니다.
    df_for_corr = X.copy()
    df_for_corr['drying_time_minutes'] = y

    plt.figure(figsize=(10, 8))

    # 만약 한글 폰트가 깨질 경우, 시스템에 설치된 한글 폰트로 설정해야 합니다.
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 예시

    corr = df_for_corr.corr()
    heatmap = sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    heatmap.set_title('건조 시간과 주요 특징 간의 상관관계 히트맵', fontsize=16, pad=12)

    plt.savefig('correlation_heatmap.png')
    print("상관관계 히트맵을 'correlation_heatmap.png' 파일로 저장했습니다.")
    plt.show()


def create_and_save_model(X, y):
    """전체 데이터로 모델을 학습하고 파일로 저장합니다."""
    if X.empty or y.empty:
        print("학습할 데이터가 없어 모델 생성을 건너뜁니다.")
        return None

    print("\n--- 모델 학습 및 저장 시작 ---")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    model.fit(X, y)

    joblib.dump(model, 'drying_model.pkl')
    print("모델을 'drying_model.pkl' 파일로 저장했습니다.")
    return model


def make_features_for_prediction(current_session_df):
    """(예측용) 현재 진행중인 세션의 데이터로 모델 입력 특징을 생성합니다."""
    if current_session_df.empty:
        return None

    # SettingWithCopyWarning을 피하기 위해 명시적으로 복사본을 생성합니다.
    df = current_session_df.copy()

    # 1. 조도 평균 계산
    light_cols = ['light_lux_0', 'light_lux_1', 'light_lux_2', 'light_lux_3']
    df['light_lux_avg'] = df[light_cols].mean(axis=1)

    # 2. 초기 옷 습도: 첫 번째 데이터의 값
    initial_cloth_humidity = df.iloc[0]['cloth_moist_pct']

    # 3. 평균 환경 값: 현재까지 쌓인 데이터의 평균
    avg_temp = df['temp_c'].mean()
    avg_humidity = df['hum_pct'].mean()
    avg_light = df['light_lux_avg'].mean()

    # 4. 모델이 학습한 순서대로 2D 배열로 만듦
    features = np.array([[
        avg_temp,
        avg_humidity,
        avg_light,
        initial_cloth_humidity
    ]])

    return features


# --- 메인 실행 로직 ---
if __name__ == '__main__':
    # --- 0. 설정 ---
    FIREBASE_KEY_PATH = "firebase.json"
    COLLECTION_PATH = "devices/DRYING01/readings"
    # 1. 건조 완료로 판단할 옷 습도 임계값 (%)
    DRYING_COMPLETE_THRESHOLD = 5.0

    # --- 1. 학습 단계 (앱 배포 전 한 번만 실행) ---
    print("--- Firestore에서 전체 학습 데이터 로드 시작 ---")
    all_completed_data = fetch_data_from_firestore(FIREBASE_KEY_PATH, COLLECTION_PATH)

    if not all_completed_data.empty:
        # --- 수정된 부분 ---
        # 시뮬레이션과 학습 모두에서 사용하기 위해 원본 데이터프레임을 미리 정렬합니다.
        all_completed_data.sort_values(by='ts', inplace=True)
        # --- 수정 끝 ---

        X, y = preprocess_data_for_training(all_completed_data)

        create_correlation_heatmap(X, y)
        create_and_save_model(X, y)
    else:
        print("Firestore에서 데이터를 가져오지 못해 학습을 진행할 수 없습니다.")

    print("\n" + "=" * 50 + "\n")

    # --- 2. 실시간 예측 단계 (실제 서비스에서 반복 실행되는 부분) ---
    print("--- 실시간 예측 시뮬레이션 시작 ---")

    try:
        loaded_model = joblib.load('drying_model.pkl')
        print("저장된 모델 'drying_model.pkl'을 불러왔습니다.")
    except FileNotFoundError:
        print("저장된 모델 파일이 없습니다. 먼저 모델을 학습시켜주세요.")
        exit()

    # 2-2. 새로운 건조 세션 시작 (시뮬레이션)
    if not all_completed_data.empty and all_completed_data.shape[0] > 15:
        # 이제 all_completed_data가 정렬되었으므로 tail(15)는 시간상 가장 최신 15개 데이터를 가져옵니다.
        new_session_data = all_completed_data.tail(15).copy().reset_index(drop=True)

        actual_time_sec = (new_session_data['ts'].max() - new_session_data['ts'].min()).total_seconds()
        print(f"\n새로운 건조 시작! (시뮬레이션용 데이터 {len(new_session_data)}개)")
        print(f"이 세션의 실제 총 건조 시간: {actual_time_sec / 60:.0f} 분")
        print(f"건조 완료 기준 습도: {DRYING_COMPLETE_THRESHOLD}%")
        print("-" * 30)

        # 2-3. 데이터가 쌓이면서 예측이 어떻게 변하는지 확인 (시뮬레이션)
        # 실제 환경에서는 데이터가 들어올 때마다 이 로직이 실행됩니다.
        for i in range(2, len(new_session_data) + 1):
            current_data_slice = new_session_data.head(i)

            elapsed_minutes = (current_data_slice['ts'].max() - current_data_slice['ts'].min()).total_seconds() / 60
            latest_humidity = current_data_slice.iloc[-1]['cloth_moist_pct']
            latest_timestamp = current_data_slice.iloc[-1]['ts']

            print(
                f"데이터 {i}개 수집 [{latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}] (경과 시간: {elapsed_minutes:.1f}분, 현재 습도: {latest_humidity:.1f}%)")

            # 기능 1: 건조 완료 여부 판단
            if latest_humidity < DRYING_COMPLETE_THRESHOLD:
                print("==> 건조 완료 기준 도달! 건조를 종료합니다.")
                break  # 시뮬레이션 종료

            # 기능 2: 남은 시간 예측 (건조가 완료되지 않았을 경우)
            prediction_features = make_features_for_prediction(current_data_slice)

            if prediction_features is not None:
                predicted_total_time = loaded_model.predict(prediction_features)[0]
                predicted_remaining_time = predicted_total_time - elapsed_minutes
                predicted_remaining_time = max(0, predicted_remaining_time)

                print(f"==> 예측 총 건조 시간: {predicted_total_time:.2f} 분 | 예상 남은 시간: {predicted_remaining_time:.2f} 분")
    else:
        print("시뮬레이션을 위한 데이터가 부족합니다.")

