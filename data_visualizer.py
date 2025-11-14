import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import platform
import seaborn as sns  # 히트맵과 스타일을 위해 임포트

"""
데이터 원본 전처리 보여주는 파일
"""



# 1. (★) RealtimeDatabaseManager로 수정
from firebase_manager import RealtimeDatabaseManager


def fetch_data_from_rtdb(key_path, db_url, data_path):
    """
    (★) Realtime Database에서 데이터를 가져와 DataFrame으로 변환합니다.
    """
    print("--- Realtime Database에서 데이터 로드 시작 ---")
    try:
        # 1. RealtimeDatabaseManager 객체 생성 (URL 전달 필수)
        rtdb_manager = RealtimeDatabaseManager(key_path, db_url)
        # 2. 데이터 조회
        df = rtdb_manager.fetch_path_as_dataframe(data_path)
    except Exception as e:
        print(f"RTDB 데이터 조회 실패: {e}")
        return pd.DataFrame()

    if df.empty:
        print("조회된 데이터가 없습니다.")
        return pd.DataFrame()

    print(f"총 {len(df)}개의 데이터 포인트를 가져왔습니다.")
    # 시간순 정렬 (firebase_manager가 이미 수행했지만, 확인 차원)
    df.sort_values(by='timestamp', inplace=True)
    return df


# 2. (★) 모델 학습 스크립트의 전처리 로직을 (시각화를 위해) 여기에 복사
def preprocess_data_for_visualization(df_original, session_threshold_hours=1):
    """
    (학습용) 원본 데이터를 "실시간 추세" 예측 모델용으로 가공합니다.
    (시각화를 위해 원본 timestamp를 포함한 전체 DataFrame을 반환합니다.)
    """
    if df_original.empty:
        return pd.DataFrame()

    df = df_original.copy()

    # --- (★) 실제 데이터 컬럼명으로 수정 ---
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
    # --- 수정 끝 ---

    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # 세션 ID 생성
    time_diff = df['timestamp'].diff().dt.total_seconds() / 3600
    df['session_id'] = (time_diff > session_threshold_hours).cumsum()

    print(f"총 {df['session_id'].nunique()}개의 건조 세션을 감지했습니다.")

    all_sessions_data = []
    for session_id in df['session_id'].unique():
        session_df = df[df['session_id'] == session_id].copy()

        # (y) 타겟 변수 계산: "남은 건조 시간"
        end_time = session_df['timestamp'].max()
        session_df['remaining_time_minutes'] = (end_time - session_df['timestamp']).dt.total_seconds() / 60

        # (X) 실시간 피처 계산: 변화량(delta)과 추세(trend)
        session_df['Δhumidity'] = session_df['cloth_humidity'].diff().fillna(0)
        session_df['Δillumination'] = session_df['light_lux_avg'].diff().fillna(0)
        session_df['humidity_trend'] = session_df['cloth_humidity'].rolling(3).mean().bfill()  # bfill()로 수정

        all_sessions_data.append(session_df)

    # 모든 세션 데이터를 다시 하나로 합침
    processed_df = pd.concat(all_sessions_data, ignore_index=True)

    # 학습에 사용할 수 있는 데이터만 필터링 (NaN 값 등 제외)
    features = [
        'ambient_temp', 'ambient_humidity', 'light_lux_avg', 'cloth_humidity',
        'Δhumidity', 'Δillumination', 'humidity_trend', 'remaining_time_minutes',
        'timestamp'  # (★) 시각화를 위해 timestamp 컬럼 유지
    ]
    target = 'remaining_time_minutes'
    processed_df = processed_df.dropna(subset=[col for col in features if col != 'timestamp'] + [target])

    print("모델 학습용 데이터 전처리 완료.")
    return processed_df


def set_korean_font():
    """운영체제에 맞는 한글 폰트 설정"""
    system_name = platform.system()
    try:
        if system_name == 'Windows':
            plt.rc('font', family='Malgun Gothic')
        elif system_name == 'Darwin':  # Mac OS
            plt.rc('font', family='AppleGothic')
        elif system_name == 'Linux':
            plt.rc('font', family='NanumGothic')
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    except Exception as e:
        print(f"경고: 한글 폰트 설정에 실패했습니다: {e}")
        if system_name == 'Linux':
            print("리눅스의 경우 'sudo apt-get install fonts-nanum*' 폰트 설치가 필요할 수 있습니다.")


# 3. (★) 원본(Raw) 데이터 시각화 함수
def plot_raw_data(df):
    """
    (수정됨) 원본 데이터프레임을 받아 실제 센서 데이터를 시각화합니다.
    """
    if df.empty:
        print("시각화할 (Raw) 데이터가 없습니다.")
        return

    print("\n--- [1/2] 원본(Raw) 데이터 시각화 시작 ---")
    set_korean_font()

    # (★) 5행 1열: 4개 습도 센서를 한 그래프에 그리기 위해
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True, num='[원본] Raw Data')
    fig.suptitle('[원본 데이터] 시간에 따른 실제 센서 값', fontsize=20, y=1.02)

    # 1. 온도
    axes[0].plot(df['timestamp'], df['temperature'], color='red', marker='.', linestyle='-', markersize=2)
    axes[0].set_ylabel('온도 (°C)')
    axes[0].set_title('주변 온도 (temperature)')
    axes[0].grid(True)

    # 2. 습도
    axes[1].plot(df['timestamp'], df['humidity'], color='blue', marker='.', linestyle='-', markersize=2)
    axes[1].set_ylabel('습도 (%)')
    axes[1].set_title('주변 습도 (humidity)')
    axes[1].grid(True)

    # 3. 조도 (★) lux1으로 수정
    axes[2].plot(df['timestamp'], df['lux1'], color='orange', marker='.', linestyle='-', markersize=2)
    axes[2].set_ylabel('조도 (lux)')
    axes[2].set_title('조도 (lux1)')
    axes[2].grid(True)

    # 4. (★) 4개 옷 습도 센서 (Raw)
    moist_cols = ['moisture_percent_1', 'moisture_percent_2', 'moisture_percent_3', 'moisture_percent_4']
    for col in moist_cols:
        axes[3].plot(df['timestamp'], df[col], label=col, marker='.', linestyle='-', markersize=2)
    axes[3].set_ylabel('옷 습도 (%)')
    axes[3].set_title('개별 옷 습도 (moisture_percent_1~4)')
    axes[3].legend()
    axes[3].grid(True)

    # 5. (★) 4개 옷 습도 센서 (Raw - 4095 값)
    moist_raw_cols = ['moisture_raw_1', 'moisture_raw_2', 'moisture_raw_3', 'moisture_raw_4']
    for col in moist_raw_cols:
        axes[4].plot(df['timestamp'], df[col], label=col, marker='.', linestyle='-', markersize=2)
    axes[4].set_ylabel('Raw ADC 값')
    axes[4].set_title('개별 옷 습도 (Raw 값, 4095=공기중)')
    axes[4].legend()
    axes[4].grid(True)

    # X축 포맷
    xfmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S')
    axes[4].xaxis.set_major_formatter(xfmt)
    plt.xlabel('시간 (Timestamp)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])


# 4. (★) 가공된(Processed) 데이터 시각화 함수
def plot_processed_data(df):
    """
    (신규) 전처리된 데이터프레임을 받아 AI 학습용 피처를 시각화합니다.
    """
    if df.empty:
        print("시각화할 (Processed) 데이터가 없습니다.")
        return

    print("\n--- [2/2] 전처리된 (Processed) 데이터 시각화 시작 ---")
    set_korean_font()

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True, num='[전처리] Processed Data')
    fig.suptitle('[전처리된 데이터] AI 학습용 피처(Feature) 변화', fontsize=20, y=1.02)

    # 1. (타겟 y) 남은 건조 시간
    axes[0].plot(df['timestamp'], df['remaining_time_minutes'], color='purple', marker='.', linestyle='-', markersize=2)
    axes[0].set_ylabel('시간 (분)')
    axes[0].set_title('타겟(y): 남은 건조 시간 (remaining_time_minutes)')
    axes[0].grid(True)

    # 2. (피처 X) 평균 옷 습도
    axes[1].plot(df['timestamp'], df['cloth_humidity'], color='green', marker='.', linestyle='-', markersize=2)
    axes[1].set_ylabel('평균 습도 (%)')
    axes[1].set_title('피처(X): 평균 옷 습도 (cloth_humidity)')
    axes[1].grid(True)

    # 3. (피처 X) 평균 조도
    axes[2].plot(df['timestamp'], df['light_lux_avg'], color='orange', marker='.', linestyle='-', markersize=2)
    axes[2].set_ylabel('조도 (lux)')
    axes[2].set_title('피처(X): 평균 조도 (light_lux_avg)')
    axes[2].grid(True)

    # 4. (피처 X) 습도 변화량 (Delta)
    sns.lineplot(data=df, x='timestamp', y='Δhumidity', ax=axes[3], color='cyan', marker='.', markersize=2)
    axes[3].set_ylabel('변화량')
    axes[3].set_title('피처(X): 옷 습도 변화량 (Δhumidity)')
    axes[3].grid(True)

    # 5. (피처 X) 습도 추세 (Trend)
    axes[4].plot(df['timestamp'], df['cloth_humidity'], color='gray', alpha=0.5, label='평균 옷 습도 (원본)', linestyle=':')
    axes[4].plot(df['timestamp'], df['humidity_trend'], color='magenta', label='습도 추세 (Rolling 3)', marker='.',
                 linestyle='-', markersize=2)
    axes[4].set_ylabel('습도 (%)')
    axes[4].set_title('피처(X): 습도 추세 (humidity_trend)')
    axes[4].legend()
    axes[4].grid(True)

    # X축 포맷
    xfmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S')
    axes[4].xaxis.set_major_formatter(xfmt)
    plt.xlabel('시간 (Timestamp)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])


# --- 메인 실행 로직 ---
if __name__ == '__main__':
    # --- (★) 설정 (실제 환경에 맞게 수정) ---
    FIREBASE_KEY_PATH = "firebase.json"
    DATABASE_URL = "https://smart-drying-rack-fe271-default-rtdb.firebaseio.com/"
    DATA_PATH = "drying-rack-readings-1"  # 분석하고 싶은 데이터 경로

    # 1. 데이터 가져오기
    df_original = fetch_data_from_rtdb(FIREBASE_KEY_PATH, DATABASE_URL, DATA_PATH)

    if not df_original.empty:
        # 2. 데이터 전처리 (AI 학습용 피처 생성)
        df_processed = preprocess_data_for_visualization(df_original.copy())  # 원본 보존을 위해 복사본 전달

        # 3. (그래프 1) 원본 데이터 시각화
        plot_raw_data(df_original)

        # 4. (그래프 2) 전처리된 데이터 시각화
        plot_processed_data(df_processed)

        # 5. 두 그래프 모두 화면에 표시
        print("\n모든 그래프를 표시합니다. (첫 번째 그래프를 닫으면 두 번째 그래프가 보일 수 있습니다.)")
        plt.show()
    else:
        print("데이터가 없어 시각화를 진행할 수 없습니다.")