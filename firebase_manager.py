import pandas as pd
import firebase_admin
from firebase_admin import credentials, db  # 'firestore' 대신 'db'를 임포트합니다.


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
            pd.DataFrame: 변환된 데이터프레임. 데이터가 없거나 오류 발생 시 빈 데이터프REPL frame을 반환합니다.
        """
        try:
            # Firestore의 '컬렉션/문서'가 아닌 '경로(path)'로 데이터를 한 번에 가져옵니다.
            ref = self.db.reference(path)
            data = ref.get()

            if not data:
                print(f"경로 '{path}'에서 데이터를 가져오지 못했습니다. 데이터가 비어있을 수 있습니다.")
                return pd.DataFrame()

            # Realtime Database는 데이터를 딕셔너리(JSON 객체)로 반환합니다.
            # (키가 push ID이고 값이 실제 데이터인 경우가 많습니다)
            # .values()를 사용하여 실제 데이터 리스트로 만듭니다.
            data_list = list(data.values())

            df = pd.DataFrame(data_list)

            # 타임스탬프 컬럼이 있다면 datetime 객체로 변환
            if 'timestamp' in df.columns:
                # Realtime Database는 보통 Unix time(초)을 숫자로 저장합니다.
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.sort_values(by='timestamp')

            print(f"'{path}' 경로에서 성공적으로 데이터를 가져와 DataFrame으로 변환했습니다.")
            return df

        except Exception as e:
            print(f"데이터 조회 중 오류 발생: {e}")
            return pd.DataFrame()


# --- 클래스 사용 예시 ---
if __name__ == '__main__':
    FIREBASE_KEY_PATH = "firebase.json"

    # (필수) Realtime Database URL
    DATABASE_URL = "https://smart-drying-rack-fe271-default-rtdb.firebaseio.com/"

    # 조회할 경로 (컬렉션 경로와 동일하게 사용)
    DATA_PATH = "devices/DRYING01/readings"

    try:
        # 1. RealtimeDatabaseManager 객체 생성 (URL 전달 필수)
        rtdb_manager = RealtimeDatabaseManager(FIREBASE_KEY_PATH, DATABASE_URL)

        # 2. 데이터 조회
        sensor_df = rtdb_manager.fetch_path_as_dataframe(DATA_PATH)

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