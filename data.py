import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import platform  # 운영체제 확인을 위해 추가
from firebase_manager import FirestoreManager  # 기존 FirestoreManager 재사용


def fetch_and_prepare_data(key_path, collection_path):
    """
    Firestore에서 데이터를 가져와 시각화에 맞게 준비합니다.
    """
    print("--- Firestore에서 데이터 로드 시작 ---")
    try:
        fs_manager = FirestoreManager(key_path)
        df = fs_manager.fetch_collection_as_dataframe(collection_path)
    except Exception as e:
        print(f"Firestore 데이터 조회 실패: {e}")
        return pd.DataFrame()

    if df.empty:
        print("조회된 데이터가 없습니다.")
        return pd.DataFrame()

    print(f"총 {len(df)}개의 데이터 포인트를 가져왔습니다.")

    # 시간순으로 데이터 정렬
    df.sort_values(by='ts', inplace=True)

    # 조도 센서 값들의 평균을 계산하여 새로운 컬럼 생성
    light_cols = ['light_lux_0', 'light_lux_1', 'light_lux_2', 'light_lux_3']
    df['light_lux_avg'] = df[light_cols].mean(axis=1)

    return df


def plot_sensor_data(df):
    """
    데이터프레임을 받아 시간에 따른 센서 데이터를 시각화합니다.
    """
    if df.empty:
        print("시각화할 데이터가 없습니다.")
        return

    print("\n--- 데이터 시각화 시작 ---")

    # --- 수정된 부분: 운영체제에 맞는 한글 폰트 설정 ---
    system_name = platform.system()

    if system_name == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif system_name == 'Darwin':  # Mac OS
        plt.rc('font', family='AppleGothic')
    elif system_name == 'Linux':
        # 리눅스의 경우 나눔고딕 폰트가 설치되어 있어야 합니다.
        try:
            plt.rc('font', family='NanumGothic')
        except:
            print("경고: 나눔고딕 폰트가 없어 한글이 깨질 수 있습니다.")
            print("터미널에서 'sudo apt-get install fonts-nanum*' 명령어로 폰트 설치 후 다시 시도해주세요.")

    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    # --- 수정 끝 ---

    # 그래프를 4개(온도, 습도, 옷 습도, 조도) 그리기 위해 4행 1열의 서브플롯을 생성합니다.
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('시간에 따른 센서 데이터 변화', fontsize=20)

    # 1. 온도 그래프
    axes[0].plot(df['ts'], df['temp_c'], color='red', marker='.', linestyle='-')
    axes[0].set_ylabel('온도 (°C)')
    axes[0].set_title('주변 온도')
    axes[0].grid(True)

    # 2. 습도 그래프
    axes[1].plot(df['ts'], df['hum_pct'], color='blue', marker='.', linestyle='-')
    axes[1].set_ylabel('습도 (%)')
    axes[1].set_title('주변 습도')
    axes[1].grid(True)

    # 3. 옷 습도 그래프
    axes[2].plot(df['ts'], df['cloth_moist_pct'], color='green', marker='.', linestyle='-')
    axes[2].set_ylabel('옷 습도 (%)')
    axes[2].set_title('옷 습도')
    axes[2].grid(True)

    # 4. 조도 그래프
    axes[3].plot(df['ts'], df['light_lux_avg'], color='orange', marker='.', linestyle='-')
    axes[3].set_ylabel('조도 (lux)')
    axes[3].set_title('평균 조도')
    axes[3].grid(True)

    # X축 포맷 설정
    # 날짜와 시간을 보기 좋게 표시합니다.
    xfmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S')
    axes[3].xaxis.set_major_formatter(xfmt)
    plt.xlabel('시간 (Timestamp)')

    # 레이아웃을 조정하고 그래프를 화면에 표시합니다.
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    print("그래프를 화면에 표시합니다.")
    plt.show()


# --- 메인 실행 로직 ---
if __name__ == '__main__':
    # --- 설정 ---
    # 이 경로들을 실제 환경에 맞게 수정해주세요.
    FIREBASE_KEY_PATH = "firebase.json"
    COLLECTION_PATH = "devices/DRYING01/readings"

    # 1. 데이터 가져오기
    sensor_data_df = fetch_and_prepare_data(FIREBASE_KEY_PATH, COLLECTION_PATH)

    # 2. 데이터 시각화
    plot_sensor_data(sensor_data_df)

