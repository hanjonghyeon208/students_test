import streamlit as st
import joblib
import numpy as np

# 모델 로드
model = joblib.load('model.pkl')

st.title("🎓 학생 성적 예측 앱")
st.write("학생의 특성을 입력하면 최종 시험 점수를 예측합니다.")

# 입력 받기
gender = st.selectbox("성별", ['남성', '여성'])
parent_edu = st.selectbox("부모 교육 수준", ['고등학교', '학사', '석사'])
internet = st.selectbox("집에서 인터넷 사용 가능 여부", ['아니오', '예'])
activities = st.selectbox("과외 활동 참여 여부", ['아니오', '예'])
study_hours = st.slider("주당 공부 시간", 0.0, 50.0, 10.0)
attendance = st.slider("출석률 (%)", 0.0, 100.0, 90.0)
past_scores = st.slider("이전 시험 평균 점수", 0.0, 100.0, 75.0)

# 라벨 인코딩 (훈련 시 순서 유지)
gender_enc = 1 if gender == '여성' else 0
parent_map = {'고등학교': 0, '학사': 1, '석사': 2}
internet_enc = 1 if internet == '예' else 0
activities_enc = 1 if activities == '예' else 0

interaction = study_hours * attendance

# 예측
features = np.array([[study_hours, attendance, past_scores, gender_enc,
                      parent_map[parent_edu], internet_enc, activities_enc,
                      interaction]])
pred = model.predict(features)[0]

st.subheader(f"📘 예측된 최종 시험 점수: **{pred:.2f}** 점")
