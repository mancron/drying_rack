# ============================================
# 📦 건조 시간 예측 AI (XGBoost 회귀)
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt

# ============================================
# 1️⃣ 데이터 불러오기
# ============================================

# 예시: 센서에서 수집한 CSV (air_temp, cloth_humidity, day_temp, day_humidity, illumination, time_to_dry)
data = pd.read_csv("dryer_data.csv")

# 데이터 확인
print("데이터 샘플:")
print(data.head())

# ============================================
# 2️⃣ 피처 엔지니어링
# ============================================

# 시간 흐름 반영용 파생 피처 생성
data['Δhumidity'] = data['cloth_humidity'].diff().fillna(0)
data['Δillumination'] = data['illumination'].diff().fillna(0)
data['humidity_trend'] = data['cloth_humidity'].rolling(3).mean().fillna(method='bfill')

# 사용 피처 목록
features = [
    'air_temp',
    'cloth_humidity',
    'illumination',
    'Δhumidity',
    'Δillumination',
    'humidity_trend'
]

target = 'time_to_dry'

# ============================================
# 3️⃣ 데이터 분할 및 스케일링
# ============================================

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 4️⃣ XGBoost 회귀 모델 학습
# ============================================

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ============================================
# 5️⃣ 예측 및 평가
# ============================================

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 예측 성능 평가:")
print(f"MAE (평균 절대 오차): {mae:.2f}분")
print(f"R² (결정계수): {r2:.3f}")

# ============================================
# 6️⃣ 중요 변수 시각화
# ============================================

plt.figure(figsize=(8,6))
plot_importance(model, importance_type='gain', title='Feature Importance')
plt.show()


# ============================================
# 7️⃣ 새 데이터로 예측 (실시간 입력 예시) - 수정 제안
# ============================================

# 💡 실시간 예측을 위한 함수 정의 (직전 시점 데이터를 인수로 받음)
def predict_dry_time(current_data, prev_data_1, prev_data_2, scaler, model, features):
    # 1. 롤링 평균 및 차분 피처 계산
    current_humidity = current_data['cloth_humidity']
    current_illumination = current_data['illumination']

    # Δhumidity 계산: 현재 - 직전_1
    delta_humidity = current_humidity - prev_data_1['cloth_humidity']
    # Δillumination 계산: 현재 - 직전_1
    delta_illumination = current_illumination - prev_data_1['illumination']
    # humidity_trend 계산: (현재 + 직전_1 + 직전_2) / 3
    humidity_trend = (current_humidity + prev_data_1['cloth_humidity'] + prev_data_2['cloth_humidity']) / 3

    # 2. 예측에 사용할 DataFrame 생성
    new_input = pd.DataFrame([{
        'air_temp': current_data['air_temp'],
        'cloth_humidity': current_humidity,
        'illumination': current_illumination,
        'Δhumidity': delta_humidity,  # 계산된 값 사용
        'Δillumination': delta_illumination,  # 계산된 값 사용
        'humidity_trend': humidity_trend  # 계산된 값 사용
    }], columns=features)

    # 3. 스케일링 및 예측
    new_input_scaled = scaler.transform(new_input)
    predicted_time = model.predict(new_input_scaled)[0]

    return predicted_time


# ----------------- 예측 실행 예시 -----------------


prev_data_1 = {'cloth_humidity': 42.3, 'illumination': 470}  # 직전 시점
prev_data_2 = {'cloth_humidity': 44.7, 'illumination': 460}  # 직전전 시점

current_data = {
    'air_temp': 26.5,
    'cloth_humidity': 40.2,  # 현재 시점
    'illumination': 480
}

predicted_time = predict_dry_time(
    current_data=current_data,
    prev_data_1=prev_data_1,
    prev_data_2=prev_data_2,
    scaler=scaler,
    model=model,
    features=features
)

print(f"\n🕒 수정된 예측 로직 (이전 데이터 기반): {predicted_time:.1f}분")
