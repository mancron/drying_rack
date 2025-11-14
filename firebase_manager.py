import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore


class FirestoreManager:
    """
    Cloud Firestore와의 연결 및 데이터 조회를 관리하는 클래스입니다.
    """

    def __init__(self, key_path):
        """
        FirestoreManager를 초기화하고 Firebase Admin SDK를 설정합니다.

        Args:
            key_path (str): Firebase 서비스 계정 키(JSON) 파일 경로
        """
        if not firebase_admin._apps:
            try:
                cred = credentials.Certificate(key_path)
                firebase_admin.initialize_app(cred)
                print("Firebase 앱이 성공적으로 초기화되었습니다.")
            except Exception as e:
                print(f"Firebase 초기화 중 오류 발생: {e}")
                raise

        # Firestore 클라이언트 인스턴스를 가져옵니다.
        self.db = firestore.client()

    def fetch_collection_as_dataframe(self, collection_path):
        """
        지정된 경로의 컬렉션(하위 컬렉션 포함) 문서를 가져와 Pandas DataFrame으로 변환합니다.

        Args:
            collection_path (str): 데이터를 가져올 Firestore의 컬렉션 전체 경로
                                 (예: 'top_collection' 또는 'top_collection/doc_id/sub_collection')

        Returns:
            pd.DataFrame: 변환된 데이터프레임. 데이터가 없거나 오류 발생 시 빈 데이터프레임을 반환합니다.
        """
        try:
            path_parts = collection_path.strip('/').split('/')
            if len(path_parts) % 2 == 0:
                print(
                    f"오류: 컬렉션 경로는 홀수 개의 세그먼트를 가져야 합니다. (예: 'collection/doc/subcollection'). 제공된 경로: '{collection_path}'")
                return pd.DataFrame()

            # 동적으로 컬렉션/문서 참조 생성
            ref = self.db
            for i, part in enumerate(path_parts):
                if i % 2 == 0:
                    ref = ref.collection(part)
                else:
                    ref = ref.document(part)

            docs = ref.stream()

            data_list = []
            for doc in docs:
                data_list.append(doc.to_dict())

            if not data_list:
                print(f"컬렉션 경로 '{collection_path}'에서 문서를 가져오지 못했습니다. 데이터가 비어있을 수 있습니다.")
                return pd.DataFrame()

            # 딕셔너리 리스트를 DataFrame으로 변환
            df = pd.DataFrame(data_list)

            # 타임스탬프 컬럼이 있다면 datetime 객체로 변환
            if 'timestamp' in df.columns:
                # Firestore 타임스탬프 또는 Unix 시간(숫자)을 가정하고 변환
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.sort_values(by='timestamp')

            print(f"'{collection_path}' 경로에서 성공적으로 데이터를 가져와 DataFrame으로 변환했습니다.")
            return df

        except Exception as e:
            print(f"데이터 조회 중 오류 발생: {e}")
            return pd.DataFrame()


# --- 클래스 사용 예시 ---
if __name__ == '__main__':
    # 실제 사용 시 아래 경로와 컬렉션 이름을 올바르게 수정해야 합니다.
    FIREBASE_KEY_PATH = "firebase.json"

    # --- 조회할 컬렉션 경로 ---
    # 1. 최상위 컬렉션 조회 예시
    # COLLECTION_PATH = "sensor_data"

    # 2. 하위 컬렉션 조회 예시 (사용자가 질문한 경로)
    COLLECTION_PATH = "devices/DRYING01/readings"

    try:
        # 1. FirestoreManager 객체 생성
        fs_manager = FirestoreManager(FIREBASE_KEY_PATH)

        # 2. 컬렉션 데이터 조회
        sensor_df = fs_manager.fetch_collection_as_dataframe(COLLECTION_PATH)

        # 3. 결과 확인
        if not sensor_df.empty:
            print("\n--- 조회된 데이터 (상위 5개) ---")
            print(sensor_df.head())
            print("\n--- 데이터 정보 ---")
            sensor_df.info()
        else:
            print("\n조회된 데이터가 없습니다.")

    except Exception as e:
        print(f"\n프로세스 실행 실패. Firebase 키 경로와 컬렉션 경로를 확인하세요: {e}")

