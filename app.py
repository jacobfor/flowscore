import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from math import pi

# ------------------------------------------------------------------------------
# 1. 설정 및 모델 로드
# ------------------------------------------------------------------------------
st.set_page_config(page_title="FlowScore AI 심사역", layout="wide")

# 한글 폰트 설정 (Mac/Window 호환)
import platform
if platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')
else: plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

@st.cache_resource
def load_model():
    return joblib.load('FlowScore_10.3_Final.pkl')

try:
    model = load_model()
except:
    st.error("❌ 모델 파일(FlowScore_10.3_Final.pkl)이 없습니다. 같은 폴더에 넣어주세요.")
    st.stop()

# ------------------------------------------------------------------------------
# 2. UI: 심사 정보 입력 (사이드바)
# ------------------------------------------------------------------------------
st.title("🏦 FlowScore AI: 기업 여신 심사 리포트")
st.markdown("---")

with st.sidebar:
    st.header("📝 기업 정보 입력")
    
    st.subheader("1. 재무/성장성 (Financial)")
    sales_curr = st.number_input("당기 매출액 (억원)", value=120.0)
    sales_prev = st.number_input("전기 매출액 (억원)", value=100.0)
    biz_score = st.slider("기업 신용평가 점수 (0~100)", 0, 100, 75)
    debt_ratio = st.number_input("부채비율 (%)", value=200.0)
    current_ratio = st.number_input("유동비율 (%)", value=120.0)
    
    st.subheader("2. 자금 활동성 (Activity)")
    late_pay_ratio = st.slider("오후(16시~) 결제 비율 (%)", 0, 100, 5)
    avg_tx_hour = st.slider("평균 결제 시간 (0~24시)", 9, 20, 14)
    avg_delay_days = st.number_input("평균 지급 지연일수 (일)", value=0.0)
    tx_volatility = st.slider("자금 변동성 (0.0~1.0)", 0.0, 1.0, 0.2)
    
    st.subheader("3. 기타 정보")
    ceo_score = st.number_input("CEO 신용점수 (KCB)", value=850)
    
    run_btn = st.button("🚀 AI 심사 실행", type="primary")

# ------------------------------------------------------------------------------
# 3. 메인 로직
# ------------------------------------------------------------------------------
if run_btn:
    # (1) 피처 엔지니어링 (모델 입력 형태로 변환)
    sales_growth = (sales_curr - sales_prev) / sales_prev if sales_prev > 0 else 0
    late_pay_rate = late_pay_ratio / 100.0
    
    # 모델 학습시 사용된 13개 변수 (순서 중요!)
    features = [
        'Biz_Score', 'Sales_Growth', 'Late_Pay_Ratio', 'Avg_Delay_Days', 
        'Debt_Ratio', 'Current_Ratio', 'Tx_Volatility', 'Avg_Tx_Hour', 
        'CEO_Score', 'Weekend_Tx_Ratio', 'OPM_Change', 'Rev_Per_Emp', 'Emp_Momentum'
    ]
    
    # 입력 데이터 구성 (일부 미입력 값은 '정상' 수준 기본값 처리)
    input_data = pd.DataFrame([{
        'Biz_Score': biz_score,
        'Sales_Growth': sales_growth,
        'Late_Pay_Ratio': late_pay_rate,
        'Avg_Delay_Days': avg_delay_days,
        'Debt_Ratio': debt_ratio,
        'Current_Ratio': current_ratio / 100.0 if current_ratio > 10 else current_ratio,
        'Tx_Volatility': tx_volatility,
        'Avg_Tx_Hour': avg_tx_hour,
        'CEO_Score': ceo_score,
        'Weekend_Tx_Ratio': 0.0, # 기본값
        'OPM_Change': 0.02,      # 기본값
        'Rev_Per_Emp': 300000,   # 기본값
        'Emp_Momentum': 0.05     # 기본값
    }])[features]

    # (2) 예측 실행
    prob = model.predict_proba(input_data)[0][1] # 승인 확률
    risk_score = (1 - prob) * 100 # 리스크 점수 (0~100)
    
    # 등급 산정
    if risk_score >= 80: grade, color = "D (위험)", "red"
    elif risk_score >= 50: grade, color = "C (경고)", "orange"
    elif risk_score >= 20: grade, color = "B (관찰)", "blue"
    else: grade, color = "A (우량)", "green"

    # (3) 결과 대시보드
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("📊 AI 종합 판정")
        st.metric(label="AI 리스크 점수", value=f"{risk_score:.1f}점", delta=grade, delta_color="inverse")
        
        if prob >= 0.5:
            st.success(f"✅ **승인 권장** (승인 확률: {prob*100:.1f}%)")
        else:
            st.error(f"🚫 **거절 권장** (부실 위험: {risk_score:.1f}%)")
            
    with col2:
        st.subheader("🧐 주요 판단 근거 (Top 3)")
        
        # 기여도 분석 (약식: 위험 요인 탐지)
        reasons = []
        if sales_growth < 0: reasons.append(f"📉 **매출 역성장**: 전년 대비 {sales_growth*100:.1f}% 감소")
        if biz_score < 60: reasons.append(f"🏢 **기업 신용 저조**: {biz_score}점 (기준 미달)")
        if late_pay_rate > 0.3: reasons.append(f"🕒 **결제 태도 불량**: 오후 결제 비중 {late_pay_ratio}% 과다")
        if avg_delay_days > 5: reasons.append(f"⚠️ **지급 지연**: 평균 {avg_delay_days}일 지연")
        if debt_ratio > 400: reasons.append(f"💰 **부채 과다**: 부채비율 {debt_ratio}%")
        
        if not reasons:
            st.info("특이한 위험 요인이 발견되지 않았습니다. 전반적으로 건전합니다.")
        else:
            for r in reasons:
                st.write(r)

    # (4) 레이더 차트 (시각화)
    st.markdown("---")
    st.subheader("🕸️ 기업 5대 역량 진단 (Radar Chart)")
    
    # 0~1 정규화 (차트용)
    # 값이 클수록 좋은 것으로 통일 (역방향 지표는 1 - value)
    data_radar = [
        min(1, max(0, (biz_score / 100))),                  # 기업신용
        min(1, max(0, (sales_growth + 0.5))),               # 성장성 (보정)
        min(1, max(0, 1 - late_pay_rate)),                  # 결제태도 (역)
        min(1, max(0, 1 - (tx_volatility))),                # 자금안정 (역)
        min(1, max(0, (ceo_score - 500)/500))               # CEO신용
    ]
    labels = ['기업신용', '성장성', '결제태도', '자금안정', 'CEO신용']
    
    # 차트 그리기
    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    data_radar += data_radar[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, data_radar, linewidth=2, linestyle='solid', color='blue')
    ax.fill(angles, data_radar, 'blue', alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)
    ax.set_yticks([]) # 눈금 숨기기
    
    st.pyplot(fig)