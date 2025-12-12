import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from math import pi
import platform

# ------------------------------------------------------------------------------
# 1. 기본 설정 및 라이브러리 로드
# ------------------------------------------------------------------------------
st.set_page_config(page_title="FlowScore AI 심사역", layout="wide")

# OpenAI 라이브러리 안전 로딩 (설치 안되어 있어도 앱이 죽지 않게)
try:
    from openai import OpenAI
    openai_installed = True
except ImportError:
    st.warning("⚠️ 'openai' 라이브러리가 설치되지 않았습니다. 터미널에 'pip install openai'를 입력하세요.")
    openai_installed = False

# 한글 폰트 설정
if platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')
else: plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------------------------
# 2. 모델 로드 (파일 없으면 가짜 모델로 테스트 가능하게 처리)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load('FlowScore_10.3_Final.pkl')

model = None
try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ 'FlowScore_10.3_Final.pkl' 파일이 없습니다! app.py와 같은 폴더에 넣어주세요.")
    # (테스트를 위해 앱이 꺼지지 않게 하려면 아래 줄 주석 처리)
    st.stop() 
except Exception as e:
    st.error(f"❌ 모델 로딩 중 에러 발생: {e}")
    st.stop()

# ------------------------------------------------------------------------------
# 3. API Key 설정 (secrets.toml 또는 직접 입력)
# ------------------------------------------------------------------------------
api_key = None
client = None

if openai_installed:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
    except:
        pass # secrets 파일 없으면 무시

    # 키가 없으면 사이드바에서 입력받기
    if not api_key:
        with st.sidebar:
            api_key = st.text_input("🔑 OpenAI API Key (미입력 시 리포트 기능 불가)", type="password")

    if api_key:
        try:
            client = OpenAI(api_key=api_key)
        except Exception as e:
            st.sidebar.error(f"API Key 오류: {e}")

# ------------------------------------------------------------------------------
# 4. 세션 상태 초기화 (화면 새로고침 방지)
# ------------------------------------------------------------------------------
if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False
if 'genai_report' not in st.session_state: st.session_state['genai_report'] = ""

# ------------------------------------------------------------------------------
# 5. 사이드바 UI (입력)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("🏦 FlowScore AI")
    st.info("기업 정보를 입력하고 하단의 실행 버튼을 눌러주세요.")
    
    st.header("1. 재무/성장성")
    sales_curr = st.number_input("당기 매출액 (억원)", value=120.0)
    sales_prev = st.number_input("전기 매출액 (억원)", value=100.0)
    biz_score = st.slider("기업 신용평가 점수", 0, 100, 75)
    debt_ratio = st.number_input("부채비율 (%)", value=200.0)
    current_ratio = st.number_input("유동비율 (%)", value=120.0)
    
    st.header("2. 자금 활동성")
    late_pay_ratio = st.slider("오후(16시~) 결제 비율 (%)", 0, 100, 5)
    avg_tx_hour = st.slider("평균 결제 시간 (0~24시)", 9, 20, 14)
    avg_delay_days = st.number_input("평균 지급 지연일수 (일)", value=0.0)
    tx_volatility = st.slider("자금 변동성 (0.0~1.0)", 0.0, 1.0, 0.2)
    
    st.header("3. 기타 정보")
    ceo_score = st.number_input("CEO 신용점수", value=850)
    
    st.markdown("---")
    # 실행 버튼
    if st.button("🚀 리스크 분석 실행", type="primary"):
        st.session_state['analyzed'] = True
        # 입력값 저장
        st.session_state['inputs'] = {
            'sales_growth': (sales_curr - sales_prev) / sales_prev if sales_prev > 0 else 0,
            'debt_ratio': debt_ratio,
            'current_ratio': current_ratio,
            'late_pay_ratio': late_pay_ratio,
            'avg_delay_days': avg_delay_days,
            'tx_volatility': tx_volatility,
            'biz_score': biz_score,
            'ceo_score': ceo_score,
            'avg_tx_hour': avg_tx_hour
        }
        # 리포트 초기화
        st.session_state['genai_report'] = ""

# ------------------------------------------------------------------------------
# 6. 메인 화면 로직
# ------------------------------------------------------------------------------
st.title("📊 기업 여신 심사 리포트")

if not st.session_state['analyzed']:
    st.info("👈 왼쪽 사이드바에서 데이터를 입력하고 [분석 실행] 버튼을 눌러주세요.")
    st.stop()

# (분석 실행 버튼이 눌린 상태라면 아래 실행)
vals = st.session_state['inputs']

# 1) 데이터 전처리
features = [
    'Biz_Score', 'Sales_Growth', 'Late_Pay_Ratio', 'Avg_Delay_Days', 
    'Debt_Ratio', 'Current_Ratio', 'Tx_Volatility', 'Avg_Tx_Hour', 
    'CEO_Score', 'Weekend_Tx_Ratio', 'OPM_Change', 'Rev_Per_Emp', 'Emp_Momentum'
]

input_df = pd.DataFrame([{
    'Biz_Score': vals['biz_score'],
    'Sales_Growth': vals['sales_growth'],
    'Late_Pay_Ratio': vals['late_pay_ratio'] / 100.0,
    'Avg_Delay_Days': vals['avg_delay_days'],
    'Debt_Ratio': vals['debt_ratio'],
    'Current_Ratio': vals['current_ratio'] / 100.0,
    'Tx_Volatility': vals['tx_volatility'],
    'Avg_Tx_Hour': vals['avg_tx_hour'],
    'CEO_Score': vals['ceo_score'],
    'Weekend_Tx_Ratio': 0.0, 'OPM_Change': 0.02, 'Rev_Per_Emp': 300000, 'Emp_Momentum': 0.05
}])[features]

# 2) 모델 예측
try:
    prob = model.predict_proba(input_df)[0][1]
    risk_score = (1 - prob) * 100
except Exception as e:
    st.error(f"모델 예측 중 오류 발생: {e}")
    st.stop()

if risk_score >= 80: grade, color = "D (위험)", "red"
elif risk_score >= 50: grade, color = "C (경고)", "orange"
elif risk_score >= 20: grade, color = "B (관찰)", "blue"
else: grade, color = "A (우량)", "green"

# 3) 결과 대시보드 출력
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("🎯 AI 종합 판정")
    st.metric(label="리스크 점수", value=f"{risk_score:.1f}점", delta=grade, delta_color="inverse")
    if prob >= 0.5:
        st.success(f"✅ **승인 권장** (승인확률 {prob*100:.1f}%)")
    else:
        st.error(f"🚫 **거절 권장** (부실위험 {risk_score:.1f}%)")

with col2:
    st.subheader("🕸️ 5대 역량 진단")
    # 레이더 차트
    data_radar = [
        min(1, max(0, (vals['biz_score'] / 100))),
        min(1, max(0, (vals['sales_growth'] + 0.5))),
        min(1, max(0, 1 - (vals['late_pay_ratio']/100))),
        min(1, max(0, 1 - (vals['tx_volatility']))),
        min(1, max(0, (vals['ceo_score'] - 500)/500))
    ]
    labels = ['기업신용', '성장성', '결제태도', '자금안정', 'CEO신용']
    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    data_radar += data_radar[:1]
    
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, data_radar, linewidth=2, linestyle='solid', color='blue')
    ax.fill(angles, data_radar, 'blue', alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_yticks([])
    st.pyplot(fig)

## ------------------------------------------------------------------------------
# 7. XAI 심층 분석 및 GenAI 보고서 생성
# ------------------------------------------------------------------------------
st.divider()
st.header("🔍 AI 심층 분석 및 상세 보고서")

# [XAI 섹션 1] 리스크 요인 분해 (Contribution Analysis)
st.subheader("1. 핵심 리스크 요인 (Key Risk Drivers)")
st.caption("AI 모델의 판단에 가장 큰 영향을 미친 긍정/부정 요인을 분석합니다.")

# 요인 분석 로직 (Rule-based로 시뮬레이션)
positives = []
negatives = []

# (1) 활동성 요인
if vals['late_pay_ratio'] > 20:
    negatives.append((f"오후결제 과다 ({vals['late_pay_ratio']}%)", "유동성 경색 징후"))
elif vals['late_pay_ratio'] < 5:
    positives.append(("건전한 결제 습관", "오후결제 5% 미만"))

if vals['avg_delay_days'] > 5:
    negatives.append((f"지급 지연 ({vals['avg_delay_days']}일)", "상환 여력 부족"))
else:
    positives.append(("성실 상환", "지연 없음"))

# (2) 재무 요인
if vals['debt_ratio'] > 300:
    negatives.append((f"부채 비율 위험 ({vals['debt_ratio']}%)", "자본 잠식 우려"))
elif vals['debt_ratio'] < 100:
    positives.append(("재무 구조 건전", "부채비율 100% 미만"))

if vals['sales_growth'] < 0:
    negatives.append(("매출 역성장", f"{vals['sales_growth']*100:.1f}% 감소"))
elif vals['sales_growth'] > 0.2:
    positives.append(("고성장세", f"{vals['sales_growth']*100:.1f}% 증가"))

# UI 출력
col_pos, col_neg = st.columns(2)

with col_pos:
    st.info("🟢 **긍정적 요인 (Positive Factors)**")
    if positives:
        for title, desc in positives:
            st.markdown(f"- **{title}**: {desc}")
    else:
        st.write("뚜렷한 긍정 요인이 부족합니다.")

with col_neg:
    st.error("🔴 **부정적 요인 (Negative Factors)**")
    if negatives:
        for title, desc in negatives:
            st.markdown(f"- **{title}**: {desc}")
    else:
        st.write("발견된 특이 리스크가 없습니다.")

# [XAI 섹션 2] 피어 그룹 비교 (Peer Comparison)
st.markdown("---")
st.subheader("2. 업계 기준 대비 위치 (Peer Comparison)")

col1, col2, col3 = st.columns(3)

# 기준값 설정 (임의의 업계 평균)
REF_LATE_PAY = 10.0  # 위험 기준 10%
REF_DEBT = 200.0     # 위험 기준 200%
REF_DELAY = 5.0      # 위험 기준 5일

with col1:
    st.write("**🕒 오후결제비율**")
    curr = vals['late_pay_ratio']
    # 0~100 사이 비율 계산 (시각화용)
    bar_val = min(1.0, curr / 50.0) 
    st.progress(bar_val)
    st.caption(f"현재 {curr}% vs 안전기준 {REF_LATE_PAY}%")
    if curr > REF_LATE_PAY: st.markdown(":red[**위험 초과**]")

with col2:
    st.write("**💰 부채비율**")
    curr = vals['debt_ratio']
    bar_val = min(1.0, curr / 500.0)
    st.progress(bar_val)
    st.caption(f"현재 {curr}% vs 안전기준 {REF_DEBT}%")
    if curr > REF_DEBT: st.markdown(":red[**위험 초과**]")

with col3:
    st.write("**⚠️ 평균지연일수**")
    curr = vals['avg_delay_days']
    bar_val = min(1.0, curr / 30.0)
    st.progress(bar_val)
    st.caption(f"현재 {curr}일 vs 안전기준 {REF_DELAY}일")
    if curr > REF_DELAY: st.markdown(":red[**위험 초과**]")

# [XAI 섹션 3] GenAI 종합 보고서
st.markdown("---")
st.subheader("3. AI 심사역 종합 의견서")

if not client:
    st.warning("⚠️ OpenAI API Key가 연결되지 않았습니다. 사이드바에 키를 입력하거나 secrets.toml을 확인하세요.")
else:
    if st.button("📄 상세 보고서 생성하기 (GPT-4o)", type="primary"):
        with st.spinner("AI 심사역이 정밀 분석 보고서를 작성 중입니다..."):
            
            # 프롬프트 고도화 (표 작성 요청 포함)
            prompt = f"""
            당신은 20년 차 베테랑 금융 심사역입니다. 다음 기업 데이터를 바탕으로 [여신 심사 보고서]를 작성하세요.
            
            [심사 개요]
            - 기업명: (주)신청기업
            - AI 예측 점수: {risk_score:.1f}점 (등급: {grade})
            - 최종 판정: {'승인 권장' if prob >= 0.5 else '거절 권장'}
            
            [상세 데이터]
            1. 재무건전성
               - 매출성장률: {vals['sales_growth']*100:.1f}% (전기 대비)
               - 부채비율: {vals['debt_ratio']}%
               - 유동비율: {vals['current_ratio']*100:.1f}%
            
            2. 활동성(FlowPoint)
               - 오후결제비율: {vals['late_pay_ratio']}% (※핵심 리스크 지표)
               - 평균지연일수: {vals['avg_delay_days']}일
               - 자금변동성: {vals['tx_volatility']}
            
            [작성 지시사항]
            1. **종합 의견**: 승인/거절 여부와 그 핵심 사유를 두괄식으로 작성할 것.
            2. **지표 상세 분석**: 
               - 재무지표와 활동성지표 간의 괴리(예: 매출은 좋은데 결제 태도가 나쁜 경우)를 중점적으로 분석할 것.
               - '오후결제비율'이 높다면 유동성 위기 가능성을 강력하게 경고할 것.
            3. **요약 테이블**: 주요 지표의 상태(양호/주의/위험)를 마크다운 표(Table)로 정리할 것.
            """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": "전문적인 금융 보고서 톤(Markdown 형식)으로 작성하세요."},
                              {"role": "user", "content": prompt}],
                    temperature=0.7
                )
                st.session_state["genai_report"] = response.choices[0].message.content
            except Exception as e:
                st.error(f"❌ 에러 발생: {e}")

# 생성된 리포트 표시
if st.session_state["genai_report"]:
    st.markdown(st.session_state["genai_report"])