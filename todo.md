# 데이터 셋 분석

- 이상데이터 제거 [Done]

- 클래스 불균형 분석 [Done]

- 해상도 분석 [Done]

# 데이터 셋 증강

- Cutmix [Done]

- RandomCrop [Done]

- CoarseDropout [Done]

- Rotate [Done]

- Curriculum learning [Done]

# 외부 모델 활용

- Detection [Done]

# Train

- K-Fold [Done]

- Model (Convnext V2, ViT, Swin V2, InternImage) [Done]

- EMA Weight Update [Done]

- AttnPool [Done]

# 추론

- TTA [Done]

- Ensemble [Done]


06월 09일까지 팀빌딩 마무리, 

6 17 대회 마감


# 증강 변경
    - Crop 0.25 vs 0.5 [Done] : 0.25가 나음
    - mean, std 127.5 : 큰차이 없음
    - BGR2RGB Apply vs no :좀 더 나은 거 같음
    - mix up
    - RGB Shift Apply vs no
    - A.GaussianBlur 추가
# 학습
    - EMA : 성능 급감 
    - Label smoothing 0.1 vs 0 : Log Loss의 특성상 성능이 더 안좋아질 가능성 있음 [Done]

# 모델
    - Liner 층 개수 0, 1, 2 (Linear->GELU) : 할 수록 성능 급감
    - Layernorm apply vs no : 성능 감소

# 추론
    -- TTA 조합