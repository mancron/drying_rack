import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform


def _set_korean_font():
    """운영체제에 맞는 한글 폰트 설정 (내부 함수)"""
    system_name = platform.system()
    try:
        if system_name == 'Windows':
            plt.rc('font', family='Malgun Gothic')
        elif system_name == 'Darwin':  # Mac OS
            plt.rc('font', family='AppleGothic')
        elif system_name == 'Linux':
            plt.rc('font', family='NanumGothic')

        # 마이너스 기호 깨짐 방지
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"경고: 한글 폰트 설정에 실패했습니다: {e}")


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

    # 한글 폰트 설정
    _set_korean_font()

    df_for_corr = X.copy()
    df_for_corr['remaining_time_minutes'] = y

    plt.figure(figsize=(10, 8))

    corr = df_for_corr.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('건조 시간과 주요 특징 간의 상관관계 히트맵', fontsize=16, pad=12)

    try:
        plt.savefig('correlation_heatmap.png')
        print("상관관계 히트맵을 'correlation_heatmap.png' 파일로 저장했습니다.")
    except Exception as e:
        print(f"히트맵 파일 저장 중 오류 발생: {e}")

    plt.show()


if __name__ == '__main__':
    print("이 스크립트는 drying_time_predictor.py에서 임포트하여 사용하는 모듈입니다.")