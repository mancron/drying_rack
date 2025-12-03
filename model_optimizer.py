import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
from bayes_opt import BayesianOptimization

# 기존 파일에서 데이터 로드 및 전처리 함수 가져오기
# (만약 함수 이름이 preprocess_data_independent_sensors라면 그에 맞게 수정하세요)
from drying_time_predictor import fetch_all_data_from_rtdb, preprocess_data_for_training

# --- 설정 ---
FIREBASE_KEY_PATH = "firebase.json"
DATABASE_URL = "https://smart-drying-rack-fe271-default-rtdb.firebaseio.com/"
BASE_DATA_PATH = "drying-rack"

# 전역 변수로 데이터 저장 (최적화 함수가 접근할 수 있게)
X_global = None
y_global = None
groups_global = None


def xgb_evaluate(max_depth, learning_rate, n_estimators, gamma, min_child_weight, subsample, colsample_bytree):
    """
    베이지안 최적화가 호출할 '목적 함수'입니다.
    주어진 하이퍼파라미터로 모델을 학습시키고, 검증 세트의 점수(R2)를 반환합니다.
    """
    global X_global, y_global, groups_global

    # 1. 정수형 변환 (베이지안 최적화는 기본적으로 실수형을 넘겨주므로 변환 필요)
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)

    # 2. 모델 생성
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,  # CPU 병렬 처리 사용
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        gamma=gamma,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42
    )

    # 3. 데이터 분할 (세션 단위 분리 - 과적합 방지 필수!)
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)

    # groups_global이 없으면 일반 분할, 있으면 그룹 분할
    if groups_global is not None:
        train_idx, test_idx = next(splitter.split(X_global, y_global, groups=groups_global))
    else:
        # 그룹 정보가 없을 경우 대비 (예외 처리)
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(list(range(len(X_global))), test_size=0.2, random_state=42)

    X_train, X_test = X_global.iloc[train_idx], X_global.iloc[test_idx]
    y_train, y_test = y_global.iloc[train_idx], y_global.iloc[test_idx]

    # 4. 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. 학습 및 평가
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    # 점수 반환 (R2 Score가 높을수록 좋음)
    # 만약 RMSE를 줄이는 게 목표라면 return -rmse 값을 반환해야 함
    score = r2_score(y_test, preds)

    # 점수가 너무 낮으면(음수) 최적화에 방해되므로 최소한의 값 처리 (선택사항)
    return max(score, -10)


def run_optimization():
    global X_global, y_global, groups_global

    print("--- 1. 데이터 로드 및 전처리 ---")
    raw_df = fetch_all_data_from_rtdb(FIREBASE_KEY_PATH, DATABASE_URL, BASE_DATA_PATH)

    if raw_df.empty:
        print("데이터가 없습니다.")
        return

    # 전처리 함수 호출 (drying_time_predictor.py에 있는 함수 사용)
    # 주의: 반환값이 4개인지 확인하세요 (X, y, features, groups)
    try:
        X_global, y_global, features, groups_global = preprocess_data_for_training(
            raw_df,
            session_threshold_hours=2.0,
            dry_threshold_percent=1.0,
            dry_stable_rows=10
        )
    except ValueError:
        print("오류: drying_time_predictor.py의 전처리 함수가 'groups'를 반환하지 않는 것 같습니다.")
        print("전처리 함수의 return 문이 'return X, y, features, groups' 형태인지 확인해주세요.")
        return

    print(f"학습 데이터 준비 완료: {len(X_global)}개 샘플")
    print("\n--- 2. 베이지안 최적화 시작 ---")

    # 탐색할 파라미터의 범위 설정
    pbounds = {
        'max_depth': (3, 10),  # 나무의 깊이 (너무 깊으면 과적합)
        'learning_rate': (0.01, 0.3),  # 학습률
        'n_estimators': (100, 1000),  # 나무의 개수
        'gamma': (0, 5),  # 가지치기 기준
        'min_child_weight': (1, 10),  # 관측치 최소 무게
        'subsample': (0.5, 1.0),  # 데이터 샘플링 비율
        'colsample_bytree': (0.5, 1.0)  # 피처 샘플링 비율
    }

    optimizer = BayesianOptimization(
        f=xgb_evaluate,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    # 최적화 실행 (init_points: 초기 랜덤 탐색 횟수, n_iter: 최적화 반복 횟수)
    # 시간이 오래 걸리면 n_iter를 줄이세요.
    optimizer.maximize(init_points=5, n_iter=20)

    print("\n" + "=" * 50)
    print("🎉 최적의 하이퍼파라미터 발견!")
    print("=" * 50)
    best_params = optimizer.max['params']

    # 정수형 파라미터는 보기 좋게 변환
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])

    for key, value in best_params.items():
        print(f"{key}: {value}")

    print(f"\n최고 R² 점수: {optimizer.max['target']:.4f}")
    print("=" * 50)
    print("이제 위 값들을 drying_time_predictor.py의 XGBRegressor() 안에 넣어주시면 됩니다.")


if __name__ == '__main__':
    run_optimization()