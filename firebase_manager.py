import pandas as pd
import firebase_admin
from firebase_admin import credentials, db

"""
파이어베이스 연결 파일
"""


class RealtimeDatabaseManager:
    """
    Firebase Realtime Database와의 연결 및 데이터 조회를 관리하는 클래스입니다.
    """

    def __init__(self, key_path, database_url):
        """
        RealtimeDatabaseManager를 초기화하고 Firebase Admin SDK를 설정합니다.

        Args:
            key_path (str): Firebase 서비스 계정 키(JSON) 파일 경로
            database_url (str): Realtime Database의 URL
        """
        if not firebase_admin._apps:
            try:
                cred = credentials.Certificate(key_path)
                # Realtime Database는 초기화 시 'databaseURL'이 필수입니다.
                firebase_admin.initialize_app(cred, {
                    'databaseURL': database_url
                })
                print("Firebase 앱이 (Realtime Database용) 성공적으로 초기화되었습니다.")
            except Exception as e:
                print(f"Firebase 초기화 중 오류 발생: {e}")
                raise

        # Firestore 클라이언트 대신 Realtime Database 루트 레퍼런스를 가져옵니다.
        self.db = db

    def fetch_path_as_dataframe(self, path):
        """
        지정된 경로(path)의 데이터를 가져와 Pandas DataFrame으로 변환합니다.

        Args:
            path (str): 데이터를 가져올 Realtime Database의 경로 (예: 'devices/DRYING01/readings')

        Returns:
            pd.DataFrame: 변환된 데이터프레임. 데이터가 없거나 오류 발생 시 빈 데이터프레임을 반환합니다.
        """
        try:
            # Firestore의 '컬렉션/문서'가 아닌 '경로(path)'로 데이터를 한 번에 가져옵니다.
            ref = self.db.reference(path)
            data = ref.get()

            if not data:
                # print(f"경로 '{path}'에서 데이터를 가져오지 못했습니다. 데이터가 비어있을 수 있습니다.")
                return pd.DataFrame()

            # Realtime Database는 데이터를 딕셔너리(JSON 객체)로 반환합니다.
            data_list = list(data.values())

            df = pd.DataFrame(data_list)

            # 타임스탬프 컬럼이 있다면 datetime 객체로 변환
            if 'timestamp' in df.columns:
                # Realtime Database는 보통 Unix time(초)을 숫자로 저장합니다.
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values(by='timestamp')

            # print(f"'{path}' 경로에서 성공적으로 데이터를 가져와 DataFrame으로 변환했습니다.")
            return df

        except Exception as e:
            print(f"데이터 조회 중 오류 발생: {e}")
            return pd.DataFrame()

    def fetch_sequential_paths_as_dataframe(self, base_path, start_index=1):
        """
        (신규 기능) base_path-N 경로(예: base_path-1, -2, ...)를 순차적으로 확인하여
        데이터를 가져오고 하나의 DataFrame으로 합칩니다. 데이터가 없는 경로가 발견되면 중지합니다.

        Args:
            base_path (str): 경로의 기본 이름 (예: 'drying-rack-reading')
            start_index (int): 시작 인덱스 (일반적으로 1)

        Returns:
            pd.DataFrame: 합쳐진 데이터프레임.
        """
        all_data = []
        i = start_index
        print(f"--- 순차적 Realtime Database 경로 조회 시작 ({base_path}-N) ---")
        while True:
            current_path = f"{base_path}-{i}"
            print(f"  경로 조회 시도: '{current_path}'")

            # 기존 fetch_path_as_dataframe 함수를 재사용
            df = self.fetch_path_as_dataframe(current_path)

            if df.empty:
                # 데이터가 없으면 중지
                print(f"  경로 '{current_path}'에서 데이터가 없어 조회를 중단합니다. (총 {i - start_index}개 경로 조회)")
                break
            else:
                print(f"  경로 '{current_path}'에서 {len(df)}개 데이터 발견.")
                # 각 경로의 데이터를 리스트에 추가
                all_data.append(df)
                i += 1

        if not all_data:
            print("조회된 데이터가 없습니다.")
            return pd.DataFrame()

        # 모든 데이터를 하나의 DataFrame으로 합칩니다.
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"총 {i - start_index}개의 경로에서 {len(combined_df)}개의 데이터를 성공적으로 합쳤습니다.")

        # 합친 후 최종 시간순 정렬
        combined_df.sort_values(by='timestamp', inplace=True)

        return combined_df


# --- 클래스 사용 예시 ---
if __name__ == '__main__':
    FIREBASE_KEY_PATH = "firebase.json"

    # (필수) Realtime Database URL
    DATABASE_URL = "https://smart-drying-rack-fe271-default-rtdb.firebaseio.com/"

    # (★) 순차적으로 조회할 경로의 기본 이름으로 변경
    BASE_DATA_PATH = "drying-rack-reading"

    try:
        # 1. RealtimeDatabaseManager 객체 생성 (URL 전달 필수)
        rtdb_manager = RealtimeDatabaseManager(FIREBASE_KEY_PATH, DATABASE_URL)

        # 2. (★) 순차적 데이터 조회 함수 호출
        sensor_df = rtdb_manager.fetch_sequential_paths_as_dataframe(BASE_DATA_PATH)

        # 3. 결과 확인
        if not sensor_df.empty:
            print("\n--- 조회된 데이터 (상위 5개) ---")
            print(sensor_df.head())
            print("\n--- 데이터 정보 ---")
            sensor_df.info()
        else:
            print("\n조회된 데이터가 없습니다.")

    except Exception as e:
        print(f"\n프로세스 실행 실패. Firebase 키 경로와 DB URL을 확인하세요: {e}")