# 🚀 Anti-Graffiti Mask2Former (Advanced RS) QGIS 플러그인 개발 노트 최종본

본 문서는 원격 탐사(Remote Sensing) 이미지를 활용하여 그래피티 및 다양한 지표 객체를 분할(Semantic Segmentation)하기 위해 개발된 QGIS 딥러닝 플러그인의 최종 개발 내역 및 아키텍처를 상세히 기록한 노트입니다.

---

## 1. 🏗️ 프로젝트 개요 및 코어 아키텍처
본 플러그인은 QGIS 환경 내에서 PyTorch 기반의 최신 딥러닝 모델들을 손쉽게 학습하고, 불확실성(Uncertainty) 기반의 신뢰도 평가까지 원스톱으로 수행할 수 있는 전문가용 툴킷입니다.

* **지원 모델 라인업 (Multi-Architecture):**
  * **Mask2Former (HuggingFace):** 최신 SOTA 모델, Bipartite Matching 기반의 마스크 분류.
  * **SegFormer (HuggingFace):** 경량화되고 효율적인 트랜스포머 기반 분할 모델.
  * **U-Net & DeepLabV3+ (SMP):** 전통적이고 안정적인 CNN 기반의 강력한 베이스라인.
* **지원 데이터 타입:** 8-bit RGB(3-Band) 영상 완벽 지원.

---

## 2. 🎯 핵심 학습(Training) 기능 및 최적화
클래스 불균형이 극심한 위성/항공 영상의 특성을 극복하기 위해 최고급 학습 기법들이 도입되었습니다.

* **초소형/희귀 객체 탐지 성능 극대화 (Focal Loss & Class Weights):**
  * `[Auto-Calculate Optimal Weights]`: 데이터셋을 스캔하여 픽셀 빈도의 역수(Inverse Frequency) 기반으로 최적의 클래스 가중치를 자동 계산.
  * `[Enable Focal Loss]`: 쉬운 배경 픽셀은 무시하고, 탐지가 어려운 객체에 Loss 페널티를 기하급수적으로 부여 (Gamma, Alpha 튜닝 지원).
* **스마트 배경(Class 0) 학습 전략 도입:**
  * 모델이 빈 공간(Background)에 엉뚱한 클래스를 칠하는 오탐(False Positive)을 막기 위해, **Class 0을 Loss 계산(학습 대상)에 명시적으로 포함**시켰습니다.
  * 단, 점수가 부풀려지는 것을 막기 위해 **최종 평가지표(mIoU) 계산 시에는 Class 0를 제외**하도록 정교하게 분리하였습니다.
* **Transfer Learning (전이 학습):** 기존에 학습된 `.pt` 가중치를 불러와 이어서 학습 가능.

---

## 3. 📊 추론(Inference) 및 딥러닝 신뢰도 분석(Analytics)
단순한 예측을 넘어, "AI가 자신의 예측을 얼마나 확신하는가?"를 평가하는 연구용 분석 기능이 탑재되었습니다.

* **Deep Ensemble 기반 불확실성 추출:**
  * 여러 개의 시드(Seed) 모델 또는 MC-Dropout을 사용하여 **BALD, Entropy, Max-Softmax, STD** 등 4가지 불확실성 지표를 생성합니다.
* **Human-in-the-Loop (HITL) 시뮬레이션:**
  * AI가 가장 헷갈려하는(불확실성이 높은) 상위 N% 픽셀을 전문가(Human)가 수정했을 때, 모델의 성능이 얼마나 급격히 향상되는지를 시뮬레이션하는 곡선(Curve) 생성.
* **다중 모델 교차 비교 (Cross-Model Comparison):**
  * 여러 번의 실험 결과(U-Net vs Mask2Former 등)를 체크박스로 다중 선택하여, 한 장의 HTML/PNG 표와 차트로 성능(mIoU, AUROC, AUSE, HITL)을 한눈에 비교합니다.

---

## 4. 🛡️ 치명적 버그 수정 및 안정성 패치 (Troubleshooting)
폐쇄망 워크스테이션 환경에서 발생했던 PyTorch와 OS 간의 치명적인 충돌들을 완벽히 해결했습니다.

1. **Windows 페이징 파일 메모리 누수 방지 (Error 1455):**
   * 현상: Epoch가 반복될수록 공유 메모리 한계 초과로 학습이 강제 종료됨.
   * 해결: DataLoader의 `num_workers=0`으로 설정하여 멀티프로세싱 오버헤드와 Shared Memory 누수를 원천 차단. 100 Epoch 이상 완주 보장.
2. **다중 GPU(DataParallel)로 인한 CUDA 메모리 오류 해결:**
   * 현상: U-Net/DeepLabV3+ 모델 학습 시 GPU가 2개 이상인 환경에서 `DataParallel` 실행 중 CUDA 비동기 오류(`illegal memory access`) 및 레이어 텐서 디바이스 불일치 충돌 발생.
   * 해결: Windows QGIS 환경의 안정성을 위해 모든 모델(HuggingFace 및 CNN 계열)에 대해 `DataParallel` 멀티 GPU 분산 처리를 비활성화하고, 단일 GPU(`cuda:0`)로만 안전하게 구동되도록 연산을 고정함.
3. **AMP(Autocast) + BCE Loss 충돌 해결:**
   * 현상: 혼합 정밀도(FP16) 환경에서 Focal Loss의 `binary_cross_entropy` 연산 시 수치 불안정 에러 발생.
   * 해결: 해당 Loss 계산 구간에만 중첩 Context Manager(`enabled=False`)를 씌워 강제로 FP32 고정밀도 연산을 수행하게 하여 우회.
4. **스마트 디렉토리 네이밍 및 비동기 프로세스 경로 동적 연동 (UX 개선):**
   * 현상: 부모 경로 역추적이 미흡하여 폴더명이 모호하게 저장되거나, `inference.py`가 생성한 타임스탬프 결과 폴더를 `plugin.py`가 추적하지 못해 QGIS 레이어 자동 로딩 및 결과 대시보드가 정상 작동하지 않음.
   * 해결: 데이터셋 이름을 역추적하여 저장하도록 폴더 구조를 통일하고, `SubprocessWorker`가 표준 출력(stdout)에서 `Output Directory` 값을 실시간 파싱하도록 설계함. 또한 최신 타임스탬프 폴더를 동적으로 매칭하는 백업 폴백 경로 매핑 코드를 구현하여 QGIS 연동을 완벽히 안정화함.
5. **단일 시드 vs 앙상블 배경 추론(Class 0) 정합성 일치 및 불확실성 왜곡 교정:**
   * 현상: 단일 시드와 앙상블 간의 배경 argmax 로직 불일치로 성능 지표의 모순이 발생하고, 전경 클래스 슬라이싱 및 재정규화로 인해 98% 확신하는 배경 픽셀의 불확실성이 비정상적으로 높게 치솟는 왜곡 발생.
   * 해결: 앙상블도 단일 시드와 동일하게 전체 클래스(0..C-1) 기반 argmax 및 임계값 체크를 수행하도록 통일하고, Entropy/BALD/Confidence 계산을 전체 확률 분포 기준으로 보정하여 배경 노이즈 및 불확실성 수학적 왜곡 해결.
6. **학습 검증(Validation) 배경 예측 오류 수정:**
   * 현상: 학습 단계 검증부에서 Class 0 예측을 강제 배제하고 전경으로 매핑하여 False Negative를 누락시킴으로써 Val mIoU 성능이 부풀려져 측정됨.
   * 해결: 검증 예측 시에도 전체 클래스 대상 argmax를 취하도록 수정하여 실시간 학습 로그에 왜곡 없는 실제 Val mIoU 지표가 기록되도록 개선.

## 5. 🛠️ JSTARS Major Revision 대응 추가 정비 (7월 17일 ~ 20일)
원격탐사 분야 최고 권위 저널인 IEEE JSTARS 심사위원들의 지적에 대응하기 위해 엔진 고도화 및 안정화 작업을 거치며 다음 사항들이 최종 구현 및 반영되었습니다.

1. **추론 신뢰도 임계값 기본값 정상화 (0.5 ➔ 0.0):**
   * 현상: 기존 이진 분류(Binary) 흔적으로 기본 설정되어 있던 `0.5` 임계값 필터링으로 인해, 다중 클래스(10~11개) 랜드커버 분류 시 대부분의 픽셀 예측이 배경(Class 0)으로 오분류 처리되는 병목이 존재함. 이로 인해 모든 모델의 mIoU가 10~15% 하락하고 AUSE 수치가 왜곡(0.1015)됨.
   * 해결: `--bg_threshold` 기본값을 `0.0`으로 정상화하여 다중 클래스 표준 Argmax를 수행하도록 수정. 모델 성능(mIoU 75~81%)과 AUSE(0.02~0.05)를 원래 논문 수준으로 성공적으로 복원함.
2. **Temperature Scaling (확률 신뢰도 보정) 탑재:**
   * 검증(val) 세트의 예측 결과를 기반으로 L-BFGS-B 최적화 알고리즘을 수행하여 최적 보정 온도 $T$를 산출하고 이를 테스트셋에 적용하여 신뢰도 다이어그램(`reliability_diagram_scaled.png`)을 그리도록 구현.
3. **공간적 독립성 및 1,000회 Bootstrap 연산:**
   * 52개 테스트 도엽 군집 단위 복원 추출 연산을 통한 95% 신뢰구간(CI) 도출 및 GG, GS, JL 권역별 개별 성능 리포팅 자동화.
4. **오차행렬 및 분류 지표 복구:**
   * 전체 테스트셋 데이터에 대한 행 기준 정규화(Recall) 혼동행렬 이미지(`confusion_matrix.png`) 자동 생성 기능 복원 및 JSON 내 주요 분류 지표(OA, Kappa, macro-F1, 클래스별 Precision/Recall/IoU) 연산 로직 완벽 복구.
5. **다중 모델 비교 시각화 논문 포맷 롤백:**
   * 비교 차트의 지표를 기존 논문 작성 구조인 `mIoU % / Uncertainty AUROC / HITL 20% Cost Efficiency (%)`로 롤백하고, 신규 및 기존 JSON 결과 구조 모두 파싱 가능한 하위 호환성을 장착함.

---

## 6. 맺음말
본 플러그인은 초기 프로토타입에서 출발하여, 연구 목적의 까다로운 딥러닝 요구사항(불확실성 검증, 클래스 불균형 해소, 다중 앙상블)을 QGIS라는 공간정보 플랫폼 위에서 완벽하게 소화해내는 **Full-Stack AI Toolkit**으로 진화했습니다. 
어떠한 폐쇄망 오프라인 환경에서도 견고하게 동작하며, 차세대 공간정보 및 환경 탐지 연구에 강력한 무기가 될 것입니다.
