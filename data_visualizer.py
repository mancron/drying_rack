import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import platform
import seaborn as sns
from typing import List
from matplotlib.widgets import TextBox, RadioButtons
import matplotlib.pyplot as plt
"""
데이터 시각화

좌우 방향키 or 원하는 페이지 오른쪽위에 입력
왼쪽위에 라디오 버튼으로 원본/전처리 데이터 확인가능
"""

# 1. RealtimeDatabaseManager 로드
from firebase_manager import RealtimeDatabaseManager


# (★) Realtime Database용 데이터 조회 함수
def fetch_all_data_from_rtdb(key_path, db_url, base_data_path):
    print("--- Realtime Database에서 전체 데이터 순차 로드 시작 ---")
    try:
        rtdb_manager = RealtimeDatabaseManager(key_path, db_url)
        df = rtdb_manager.fetch_sequential_paths_as_dataframe(base_data_path)
    except Exception as e:
        print(f"RTDB 데이터 조회 실패: {e}")
        return pd.DataFrame()

    if df.empty:
        print("조회된 데이터가 없습니다.")
        return pd.DataFrame()

    print(f"총 {len(df)}개의 데이터 포인트를 가져왔습니다.")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by='timestamp', inplace=True)
    return df


# 2. 전처리 함수
def preprocess_data_for_visualization(df_original,
                                      session_threshold_hours=1,
                                      dry_threshold_percent=20.0,
                                      dry_stable_rows=10):
    if df_original.empty:
        return pd.DataFrame()

    df = df_original.copy()

    # --- 실제 데이터 컬럼명으로 수정 ---
    df['light_lux_avg'] = df['lux1']
    moist_cols = ['moisture_percent_1', 'moisture_percent_2', 'moisture_percent_3', 'moisture_percent_4']
    df['cloth_humidity'] = df[moist_cols].mean(axis=1)
    df = df.rename(columns={
        'temperature': 'ambient_temp',
        'humidity': 'ambient_humidity'
    })

    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # 세션 ID 생성
    time_diff = df['timestamp'].diff().dt.total_seconds() / 3600
    df['session_id'] = (time_diff > session_threshold_hours).cumsum()

    print(f"총 {df['session_id'].nunique()}개의 건조 세션을 감지했습니다.")
    print(f" (★) 건조 완료 기준: 습도 < {dry_threshold_percent}%가 {dry_stable_rows}개 포인트 연속 유지 시")

    all_sessions_data = []
    for session_id in df['session_id'].unique():
        session_df = df[df['session_id'] == session_id].copy()

        # --- "진짜" 건조 완료 시점 탐지 ---
        is_dry = session_df['cloth_humidity'] < dry_threshold_percent
        is_stable_dry = is_dry.rolling(window=dry_stable_rows).sum() >= dry_stable_rows
        stable_indices_loc = np.where(is_stable_dry)[0]

        if len(stable_indices_loc) > 0:
            first_stable_end_iloc = stable_indices_loc[0]
            first_stable_start_iloc = first_stable_end_iloc - dry_stable_rows + 1
            true_end_timestamp = session_df.iloc[first_stable_start_iloc]['timestamp']
            print(f"  (세션 {session_id}) '진짜' 건조 완료 시점 감지: {true_end_timestamp}")
            session_df = session_df[session_df['timestamp'] <= true_end_timestamp].copy()
        else:
            print(f"  (세션 {session_id}) 안정된 건조 상태를 감지하지 못함. 세션의 마지막 시간을 사용합니다.")

        end_time = session_df['timestamp'].max()
        session_df['remaining_time_minutes'] = (end_time - session_df['timestamp']).dt.total_seconds() / 60

        session_df['Δhumidity'] = session_df['cloth_humidity'].diff().fillna(0)
        session_df['Δillumination'] = session_df['light_lux_avg'].diff().fillna(0)
        session_df['humidity_trend'] = session_df['cloth_humidity'].rolling(3).mean().bfill()

        all_sessions_data.append(session_df)

    processed_df = pd.concat(all_sessions_data, ignore_index=True)

    features = [
        'ambient_temp', 'ambient_humidity', 'light_lux_avg', 'cloth_humidity',
        'Δhumidity', 'Δillumination', 'humidity_trend', 'remaining_time_minutes',
        'timestamp', 'session_id'
    ]
    check_cols = [col for col in features if col not in ['timestamp', 'session_id']]
    processed_df = processed_df.dropna(subset=check_cols)

    print("모델 학습용 데이터 전처리 완료.")
    return processed_df


def set_korean_font():
    system_name = platform.system()
    try:
        if system_name == 'Windows':
            plt.rc('font', family='Malgun Gothic')
        elif system_name == 'Darwin':
            plt.rc('font', family='AppleGothic')
        elif system_name == 'Linux':
            plt.rc('font', family='NanumGothic')
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"경고: 한글 폰트 설정에 실패했습니다: {e}")


# 3. 시각화 함수들
def plot_raw_data(axes: List[plt.Axes], df, session_id):
    if df.empty: return
    axes[0].plot(df['timestamp'], df['temperature'], color='red', marker='.', linestyle='-', markersize=2)
    axes[0].set_title('주변 온도 (temperature)', fontsize=10)
    axes[0].grid(True)

    axes[1].plot(df['timestamp'], df['humidity'], color='blue', marker='.', linestyle='-', markersize=2)
    axes[1].set_title('주변 습도 (humidity)', fontsize=10)
    axes[1].grid(True)

    axes[2].plot(df['timestamp'], df['lux1'], color='orange', marker='.', linestyle='-', markersize=2)
    axes[2].set_title('조도 (lux1)', fontsize=10)
    axes[2].grid(True)

    moist_cols = ['moisture_percent_1', 'moisture_percent_2', 'moisture_percent_3', 'moisture_percent_4']
    for col in moist_cols:
        axes[3].plot(df['timestamp'], df[col], label=col, marker='.', linestyle='-', markersize=2)
    axes[3].set_title('개별 옷 습도 (moisture_percent_1~4)', fontsize=10)
    axes[3].legend(loc='upper right', fontsize=8)
    axes[3].grid(True)

    moist_raw_cols = ['moisture_raw_1', 'moisture_raw_2', 'moisture_raw_3', 'moisture_raw_4']
    for col in moist_raw_cols:
        axes[4].plot(df['timestamp'], df[col], label=col, marker='.', linestyle='-', markersize=2)
    axes[4].set_title('개별 옷 습도 (Raw 값)', fontsize=10)
    axes[4].legend(loc='upper right', fontsize=8)
    axes[4].grid(True)

    xfmt = mdates.DateFormatter('%H:%M')
    axes[4].xaxis.set_major_formatter(xfmt)


def plot_processed_data(axes: List[plt.Axes], df, session_id):
    if df.empty: return
    axes[0].plot(df['timestamp'], df['remaining_time_minutes'], color='purple', marker='.', linestyle='-', markersize=2)
    axes[0].set_title('타겟: 남은 시간 (분)', fontsize=10)
    axes[0].grid(True)

    axes[1].plot(df['timestamp'], df['cloth_humidity'], color='green', marker='.', linestyle='-', markersize=2)
    axes[1].set_title('피처: 평균 옷 습도 (%)', fontsize=10)
    axes[1].grid(True)

    axes[2].plot(df['timestamp'], df['light_lux_avg'], color='orange', marker='.', linestyle='-', markersize=2)
    axes[2].set_title('피처: 평균 조도 (lux)', fontsize=10)
    axes[2].grid(True)

    sns.lineplot(data=df, x='timestamp', y='Δhumidity', ax=axes[3], color='cyan', marker='.', markersize=2)
    axes[3].set_title('피처: 습도 변화량', fontsize=10)
    axes[3].grid(True)

    axes[4].plot(df['timestamp'], df['cloth_humidity'], color='gray', alpha=0.5, linestyle=':')
    axes[4].plot(df['timestamp'], df['humidity_trend'], color='magenta', marker='.', linestyle='-', markersize=2)
    axes[4].set_title('피처: 습도 추세', fontsize=10)
    axes[4].grid(True)

    xfmt = mdates.DateFormatter('%H:%M')
    axes[4].xaxis.set_major_formatter(xfmt)




class SessionNavigator:
    def __init__(self, raw_data_dict, processed_data_dict, valid_sessions):
        self.raw_data_dict = raw_data_dict
        self.processed_data_dict = processed_data_dict
        self.sessions = valid_sessions
        self.current_index = 0
        self.num_sessions = len(valid_sessions)
        self.view_mode = 'Raw Data'  # 기본 보기 모드

        if self.num_sessions == 0:
            print("시각화할 세션이 없습니다.")
            return

        # 1. Figure 생성
        # 슬라이더가 없어졌으므로 상단 여백(top)을 조금 줄여도 됩니다.
        self.fig, self.axes = plt.subplots(5, 1, figsize=(16, 12), num='건조 세션 시각화')
        plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.4, left=0.1, right=0.9)

        # 2. (슬라이더 제거됨)

        # 3. 텍스트 검색 위젯 (상단 중앙~우측 배치)
        # 슬라이더 자리가 비었으니 검색창을 좀 더 잘 보이는 곳(중앙 우측)으로 옮깁니다.
        # [left, bottom, width, height]
        ax_box = plt.axes([0.65, 0.94, 0.15, 0.03])
        self.text_box = TextBox(ax_box, 'Session ID 이동: ', initial=str(self.sessions[0]))
        self.text_box.on_submit(self.submit_text)

        # 4. 보기 모드 라디오 버튼 (상단 좌측)
        ax_radio = plt.axes([0.05, 0.92, 0.12, 0.06], facecolor='#f0f0f0')
        self.radio = RadioButtons(ax_radio, ('Raw Data', 'Processed Data'))
        self.radio.on_clicked(self.change_view_mode)

        # 5. 키보드 연결
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.update_view()

        # (★) 키보드 포커스 자동 설정 (Tkinter 등 환경 호환)
        try:
            self.fig.canvas.get_tk_widget().focus_set()
        except AttributeError:
            pass

        plt.show()

    def change_view_mode(self, label):
        """라디오 버튼 클릭 시 호출"""
        self.view_mode = label
        self.update_view()

    def submit_text(self, text):
        """텍스트 박스 입력 엔터 시 호출"""
        try:
            target_id = int(text)
            if target_id in self.sessions:
                self.current_index = self.sessions.index(target_id)
                self.update_view()  # 슬라이더 업데이트 부분 삭제됨
            else:
                print(f"세션 ID {target_id}를 찾을 수 없습니다.")
        except ValueError:
            print("유효한 숫자를 입력하세요.")

    def update_view(self):
        session_id = self.sessions[self.current_index]

        # 화면 초기화
        for ax in self.axes:
            ax.clear()

        # 제목 업데이트
        self.fig.suptitle(
            f'[{self.current_index + 1}/{self.num_sessions}] 세션 {session_id} - {self.view_mode}',
            fontsize=18, y=0.98
        )

        # 모드에 따라 그래프 그리기
        if self.view_mode == 'Raw Data':
            raw_df = self.raw_data_dict[session_id]
            plot_raw_data(self.axes, raw_df, session_id)
            self.axes[0].text(0.0, 1.05, '■ 원본 센서 데이터', transform=self.axes[0].transAxes,
                              fontsize=12, weight='bold', color='blue')
        else:
            processed_df = self.processed_data_dict[session_id]
            plot_processed_data(self.axes, processed_df, session_id)
            self.axes[0].text(0.0, 1.05, '■ 학습용 데이터 (Processed)', transform=self.axes[0].transAxes,
                              fontsize=12, weight='bold', color='green')

        # X축 라벨 정리
        for ax in self.axes[:-1]:
            ax.set_xticklabels([])

        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        """키보드 조작 로직"""
        if event.key is None:
            return

        # 텍스트 박스 입력 중일 때는 그래프 이동 막기
        if event.inaxes == self.text_box.ax:
            return

        # 방향키 로직 (슬라이더 업데이트 코드 삭제됨)
        if event.key in ['right', 'n', 'down']:
            self.current_index = (self.current_index + 1) % self.num_sessions
            self.text_box.set_val(str(self.sessions[self.current_index]))  # 텍스트박스 숫자도 업데이트
            self.update_view()

        elif event.key in ['left', 'p', 'up']:
            self.current_index = (self.current_index - 1 + self.num_sessions) % self.num_sessions
            self.text_box.set_val(str(self.sessions[self.current_index]))  # 텍스트박스 숫자도 업데이트
            self.update_view()

# --- 메인 실행 ---
if __name__ == '__main__':
    set_korean_font()
    FIREBASE_KEY_PATH = "firebase.json"
    DATABASE_URL = "https://smart-drying-rack-fe271-default-rtdb.firebaseio.com/"
    BASE_DATA_PATH = "drying-rack-reading"

    df_original = fetch_all_data_from_rtdb(FIREBASE_KEY_PATH, DATABASE_URL, BASE_DATA_PATH)

    if not df_original.empty:
        df_processed = preprocess_data_for_visualization(
            df_original.copy(),
            session_threshold_hours=1.0,
            dry_threshold_percent=1.0,
            dry_stable_rows=10
        )

        df_original = df_original.sort_values(by='timestamp').reset_index(drop=True)
        time_diff = df_original['timestamp'].diff().dt.total_seconds() / 3600
        df_original['session_id'] = (time_diff > 1.0).cumsum()

        valid_sessions = sorted(df_processed['session_id'].unique())

        raw_data_dict = {sid: df_original[df_original['session_id'] == sid] for sid in valid_sessions}
        processed_data_dict = {sid: df_processed[df_processed['session_id'] == sid] for sid in valid_sessions}

        if valid_sessions:
            print(f"\n총 {len(valid_sessions)}개의 세션을 시각화합니다.")
            print("==> 상단 슬라이더를 드래그하거나, ID를 입력하거나, 좌우 키를 사용하세요. <==")
            navigator = SessionNavigator(raw_data_dict, processed_data_dict, valid_sessions)
        else:
            print("유효한 세션이 없습니다.")
    else:
        print("데이터가 없습니다.")