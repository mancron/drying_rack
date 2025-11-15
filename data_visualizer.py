import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import platform
import seaborn as sns  # 히트맵과 스타일을 위해 임포트

"""
데이터 원본 전처리 및 세션별 시각화 파일
- Firebase RTDB에서 데이터를 로드합니다.
- 데이터를 건조 세션별로 분리합니다.
- 세션별로 (1) 원본 데이터, (2) 전처리된 데이터를 시각화합니다.
- 그래프 창을 닫으면 다음 세션의 그래프가 나타납니다.
"""

# 1. (★) RealtimeDatabaseManager로 수정
# (참고: firebase_manager.py 파일이 동일한 디렉터리에 있어야 합니다)
try:
    from firebase_manager import RealtimeDatabaseManager
except ImportError:
    print("경고: firebase_manager.py를 찾을 수 없습니다.")
    print("테스트를 위해 RealtimeDatabaseManager를 임시로 정의합니다.")


    # firebase_manager.py가 없어도 실행 가능하도록 임시 클래스 정의
    class RealtimeDatabaseManager:
        def __init__(self, key_path, db_url):
            print(f"임시 RTDB 매니저 초기화 (Key: {key_path}, URL: {db_url})")
            print("실제 데이터를 가져오려면 firebase_manager.py가 필요합니다.")
            self.db_url = db_url
            self.key_path = key_path

        def fetch_path_as_dataframe(self, data_path):
            print(f"임시 데이터프레임을 생성합니다 (경로: {data_path}).")
            # 실제 실행을 위해 샘플 데이터를 생성합니다.
            base_time = pd.Timestamp.now() - pd.Timedelta(hours=5)
            # 세션 0 (2시간)
            session_0_times = pd.date_range(start=base_time, periods=120, freq='T')
            session_0_data = {
                'timestamp': session_0_times,
                'lux1': np.linspace(1000, 800, 120) + np.random.rand(120) * 50,
                'moisture_percent_1': np.linspace(90, 10, 120) + np.random.rand(120) * 5,
                'moisture_percent_2': np.linspace(95, 12, 120) + np.random.rand(120) * 5,
                'moisture_percent_3': np.linspace(88, 11, 120) + np.random.rand(120) * 5,
                'moisture_percent_4': np.linspace(92, 9, 120) + np.random.rand(120) * 5,
                'moisture_raw_1': np.linspace(1000, 4000, 120),
                'moisture_raw_2': np.linspace(900, 4000, 120),
                'moisture_raw_3': np.linspace(1100, 4000, 120),
                'moisture_raw_4': np.linspace(1050, 4000, 120),
                'temperature': np.linspace(25, 27, 120) + np.random.rand(120) * 0.5,
                'humidity': np.linspace(50, 45, 120) + np.random.rand(120) * 1,
            }
            # 세션 1 (1.5시간) - 2시간 공백 후
            session_1_start = base_time + pd.Timedelta(hours=4)
            session_1_times = pd.date_range(start=session_1_start, periods=90, freq='T')
            session_1_data = {
                'timestamp': session_1_times,
                'lux1': np.linspace(500, 400, 90) + np.random.rand(90) * 50,
                'moisture_percent_1': np.linspace(80, 15, 90) + np.random.rand(90) * 5,
                'moisture_percent_2': np.linspace(85, 18, 90) + np.random.rand(90) * 5,
                'moisture_percent_3': np.linspace(78, 16, 90) + np.random.rand(90) * 5,
                'moisture_percent_4': np.linspace(82, 14, 90) + np.random.rand(90) * 5,
                'moisture_raw_1': np.linspace(1200, 3800, 90),
                'moisture_raw_2': np.linspace(1100, 3800, 90),
                'moisture_raw_3': np.linspace(1300, 3800, 90),
                'moisture_raw_4': np.linspace(1150, 3800, 90),
                'temperature': np.linspace(22, 23, 90) + np.random.rand(90) * 0.5,
                'humidity': np.linspace(60, 58, 90) + np.random.rand(90) * 1,
            }
            df0 = pd.DataFrame(session_0_data)
            df1 = pd.DataFrame(session_1_data)
            df = pd.concat([df0, df1], ignore_index=True)
            return df


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
    # fetch_path_as_dataframe이 timestamp 변환을 보장해야 함
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by='timestamp', inplace=True)
    return df


# 2. (★) 모델 학습 스크립트의 전처리 로직을 (시각화를 위해) 여기에 복사
def preprocess_data_for_visualization(df_original,
                                      session_threshold_hours=1,
                                      dry_threshold_percent=20.0,  # (★) 건조 완료로 간주할 습도 임계값 (예: 20%)
                                      dry_stable_rows=10):  # (★) 이 습도가 유지되어야 하는 데이터 포인트 수 (예: 10분 = 10개)
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
    print(f" (★) 건조 완료 기준: 습도 < {dry_threshold_percent}%가 {dry_stable_rows}개 포인트 연속 유지 시")

    all_sessions_data = []
    for session_id in df['session_id'].unique():
        session_df = df[df['session_id'] == session_id].copy()

        # (★) --- "진짜" 건조 완료 시점 탐지 (제안하신 로직) --- (★)

        # 1. 습도가 임계값 미만인지 T/F로 표시
        is_dry = session_df['cloth_humidity'] < dry_threshold_percent

        # 2. T(1)가 dry_stable_rows 만큼 연속되는지 확인 (rolling sum)
        #    (예: 10개 윈도우의 합이 10이면, 10개 모두 True)
        is_stable_dry = is_dry.rolling(window=dry_stable_rows).sum() >= dry_stable_rows

        # 3. 이 조건을 만족하는 *첫 번째* 데이터 포인트의 인덱스(iloc)를 찾음
        stable_indices_loc = np.where(is_stable_dry)[0]  # .iloc 위치

        if len(stable_indices_loc) > 0:
            # 4. "안정"이 확인된 첫 번째 데이터 포인트의 위치 (예: 120번째)
            first_stable_end_iloc = stable_indices_loc[0]

            # 5. "안정"이 *시작*된 데이터 포인트의 위치 (예: 120 - 10 + 1 = 111번째)
            first_stable_start_iloc = first_stable_end_iloc - dry_stable_rows + 1

            # 6. "진짜 건조 완료" 시점의 타임스탬프 (111번째 데이터의 시간)
            true_end_timestamp = session_df.iloc[first_stable_start_iloc]['timestamp']

            print(f"  (세션 {session_id}) '진짜' 건조 완료 시점 감지: {true_end_timestamp}")

            # 7. (★) "0인 부분(유휴 데이터)을 자름"
            #    이 시간 이후의 데이터를 세션에서 제외
            session_df = session_df[session_df['timestamp'] <= true_end_timestamp].copy()

        else:
            # 안정된 건조 상태를 찾지 못함 (데이터가 짧거나, 건조가 안 끝남)
            print(f"  (세션 {session_id}) 안정된 건조 상태를 감지하지 못함. 세션의 마지막 시간을 사용합니다.")
        # (★) --- 로직 수정 끝 --- (★)

        # (y) 타겟 변수 계산: "남은 건조 시간"
        # (참고: end_time은 이제 '진짜' 건조 완료 시점이 됨)
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
        'timestamp',  # (★) 시각화를 위해 timestamp 컬럼 유지
        'session_id'  # (★) 세션 구분을 위해 session_id 유지
    ]
    target = 'remaining_time_minutes'
    # session_id와 timestamp를 제외한 피처 + 타겟에 대해 dropna 수행
    check_cols = [col for col in features if col not in ['timestamp', 'session_id']]
    processed_df = processed_df.dropna(subset=check_cols)

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
def plot_raw_data(df, session_id=None):  # (★) session_id 인자 추가
    """
    (수정됨) 원본 데이터프레임을 받아 실제 센서 데이터를 시각화합니다.
    """
    if df.empty:
        print(f"시각화할 (Raw) 데이터가 없습니다. (세션 {session_id})")
        return

    print(f"\n--- [1/2] 세션 {session_id} 원본(Raw) 데이터 시각화 시작 ---")
    set_korean_font()

    # (★) Figure 번호와 제목에 세션 ID 반영
    fig_num = f'[세션 {session_id}] Raw' if session_id is not None else '[전체] Raw Data'
    fig_title = f'{fig_num}: 시간에 따른 실제 센서 값'

    # (★) 5행 1열: 4개 습도 센서를 한 그래프에 그리기 위해
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True, num=fig_num)  # num=fig_num
    fig.suptitle(fig_title, fontsize=20, y=0.98)  # (★) y=1.02에서 0.98로 내려서 보이도록 수정

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
def plot_processed_data(df, session_id=None):  # (★) session_id 인자 추가
    """
    (신규) 전처리된 데이터프레임을 받아 AI 학습용 피처를 시각화합니다.
    """
    if df.empty:
        print(f"시각화할 (Processed) 데이터가 없습니다. (세션 {session_id})")
        return

    print(f"\n--- [2/2] 세션 {session_id} 전처리된 (Processed) 데이터 시각화 시작 ---")
    set_korean_font()

    # (★) Figure 번호와 제목에 세션 ID 반영
    fig_num = f'[세션 {session_id}] Processed' if session_id is not None else '[전체] Processed Data'
    fig_title = f'{fig_num}: AI 학습용 피처(Feature) 변화'

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True, num=fig_num)  # num=fig_num
    fig.suptitle(fig_title, fontsize=20, y=0.98)  # (★) y=1.02에서 0.98로 내려서 보이도록 수정

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
    DATA_PATH = "drying-rack-reading-1"  # 분석하고 싶은 데이터 경로
    SESSION_THRESHOLD_HOURS = 1.0  # (★) 세션 분리 기준 시간 (1시간 이상 데이터 없으면 새 세션)

    # (★) --- 새 파라미터 --- (★)
    # (이 값을 조절하여 '건조 완료' 시점을 튜닝할 수 있습니다)
    DRY_THRESHOLD = 1.0  # 건조 완료로 간주할 습도 (예: 20%)
    DRY_STABLE_POINTS = 10  # 위 습도가 연속으로 유지되어야 하는 데이터 개수 (예: 10개 = 10분)
    # (★) --- --- (★)

    # 1. 데이터 가져오기
    df_original = fetch_data_from_rtdb(FIREBASE_KEY_PATH, DATABASE_URL, DATA_PATH)

    if not df_original.empty:
        # 2. 데이터 전처리 (AI 학습용 피처 생성)
        df_processed = preprocess_data_for_visualization(
            df_original.copy(),  # 원본 보존을 위해 복사본 전달
            session_threshold_hours=SESSION_THRESHOLD_HOURS,
            dry_threshold_percent=DRY_THRESHOLD,  # (★) 전달
            dry_stable_rows=DRY_STABLE_POINTS  # (★) 전달
        )

        # 3. (★) 원본 데이터에도 세션 ID 추가 (필터링을 위함)
        #    (전처리 함수가 원본 복사본을 사용하므로, 원본에도 수동 할당 필요)
        print(f"원본 데이터에 세션 ID 할당 (기준: {SESSION_THRESHOLD_HOURS}시간)...")
        df_original = df_original.sort_values(by='timestamp').reset_index(drop=True)
        time_diff = df_original['timestamp'].diff().dt.total_seconds() / 3600
        df_original['session_id'] = (time_diff > SESSION_THRESHOLD_HOURS).cumsum()

        # 4. (★) 유효한 세션 ID 목록 가져오기 (전처리 후 남은 세션)
        valid_sessions = df_processed['session_id'].unique()
        valid_sessions.sort()

        if len(valid_sessions) == 0:
            print("전처리 후 유효한 세션이 남아있지 않습니다. (데이터가 너무 짧을 수 있음)")
        else:
            print(f"총 {len(valid_sessions)}개의 유효한 세션을 순차적으로 시각화합니다.")
            print("그래프 창을 닫으면 다음 세션 그래프가 나타납니다.")

            # 5. (★) 세션별로 반복하며 그래프 표시
            for session_id in valid_sessions:
                print(f"\n--- 세션 ID {session_id} 표시 ---")

                # 이 세션에 해당하는 데이터 필터링
                session_raw_df = df_original[df_original['session_id'] == session_id]
                session_processed_df = df_processed[df_processed['session_id'] == session_id]

                if session_processed_df.empty:
                    print(f"(세션 {session_id}는 전처리 과정에서 제외되어 스킵합니다.)")
                    continue

                # (그래프 1) 원본 데이터 시각화
                plot_raw_data(session_raw_df, session_id)

                # (그래프 2) 전처리된 데이터 시각화
                plot_processed_data(session_processed_df, session_id)

                # (★) plt.show()를 루프 안에 넣어 "페이지 넘기기" 효과 구현
                # 사용자가 이 세션의 그래프 2개를 모두 닫아야 다음 루프가 실행됨
                print(f"세션 {session_id}의 그래프 2개를 표시합니다. 창을 닫으면 다음 세션으로 넘어갑니다...")
                plt.show()

        print("\n모든 세션 시각화 완료.")
    else:
        print("데이터가 없어 시각화를 진행할 수 없습니다.")