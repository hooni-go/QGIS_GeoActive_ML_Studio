# 📘 Advanced RS Mask2Former Plugin - 사용자 매뉴얼 (User Manual)

본 매뉴얼은 딥러닝이나 코딩을 잘 모르는 연구원 및 사용자도 QGIS 내에서 원격탐사(Remote Sensing) 위성/항공 이미지를 손쉽게 학습하고, 추론 결과를 분석하며, 논문 작성을 위한 불확실성 평가 및 교차 모델 벤치마킹을 원클릭으로 수행할 수 있도록 단계별 가이드를 제공합니다.

---

## 🌟 주요 핵심 기능 요약
1. **4대 인공지능 세그멘테이션 모델 지원**: `Mask2Former`, `U-Net`, `DeepLabV3+`, `SegFormer`
2. **다중 시드(Multi-Seed) 및 MC Dropout**: 1개 모델만으로도 다중 시드 예측을 수행하여 신뢰성 높은 **불확실성 지도(Uncertainty Map)** 추출
3. **학술 논문용 분석 차트 자동 생성**: Sparsification Plot, HITL 비용 절감 곡선, 오분류 탐지 AUROC, Cohen's Kappa, ECE 지표 등 산출
4. **오류 차단 장치 & 논문용 사본 저장**: 모델 불일치 자동 감지 및 스마트 스위치, 논문 복사용 모델 접두사(Prefix) 사본 자동 저장

---

## 🛠️ Step 0. 기본 환경 설정
플러그인을 실행하면 QGIS 좌측/우측 또는 상단 아이콘을 통해 플러그인 메인 다이얼로그 창을 띄울 수 있습니다.

1. **Portable Env Dir (파이썬 가상환경)**: 
   * 플러그인 폴더 내의 포터블 파이썬 가상환경(`python_env`)이 **자동 감지 및 로드**됩니다. 별도의 파이썬 설치나 라이브러리 세팅이 불필요합니다.
2. **Model Architecture (모델 선택)**:
   * 글로벌 설정으로 사용할 모델(Mask2Former, U-Net, DeepLabV3+, SegFormer)을 드롭다운에서 선택합니다.

---

## 🚀 Step 1. 모델 학습하기 (Training)

새로운 원격탐사 데이터를 인공지능에게 가르치는 단계입니다.

1. **[Training]** 탭을 선택합니다.
2. **Dataset Root (데이터셋 경로)**:
   * `[Browse]` 버튼을 눌러 학습 데이터셋의 루트 폴더를 지정합니다. 
   * *구조 안내:* 선택한 폴더 내부에 `train/image`, `train/label`, `val/image`, `val/label` 폴더 구조가 기본적으로 존재해야 합니다.
3. **Data Type (데이터 형식)**:
   * 일반 8비트 항공 영상은 **`Aerial 3-Band (RGB, 8-bit)`**를 선택합니다.
   * 16비트 위성 영상 및 NIR 밴드가 포함된 영상은 **`Satellite 4-Band (RGBN, 16-bit)`**를 선택합니다. (라벨 tiff 파일은 `rasterio` 라이브러리를 통해 16-bit 정보 훼손 없이 무손실로 자동 로드됩니다.)
4. **Class Mapping (클래스 맵핑)**:
   * 학습 대상이 될 클래스 번호(ID)와 색상(RGB), 그리고 실제 라벨 영상 내부의 픽셀값(DN)을 테이블에 입력합니다. (예: 0번 배경-DN 100, 1번 낙서-DN 10 등)
5. **Auto-Calculate Weights (클래스 가중치 계산)**:
   * `[🔄 Auto-Calculate Class Weights]` 버튼을 클릭하면, 학습 데이터 내 클래스 비율 불균형을 자동 분석하여 학습 손실함수에 적용할 가중치를 보정해 줍니다. (오폭 및 성능 저하 방지 필수 코스)
6. **Hyperparameters (하이퍼파라미터)**:
   * `Epochs`(학습 횟수) 및 `Batch Size`(배치 크기)를 조절합니다. (GPU 메모리가 부족할 경우 Batch Size를 2 이하로 줄여주세요.)
   * 배경이 너무 넓고 객체가 미세해 잘 잡히지 않을 경우 **`[☑️ Use Focal Loss]`**를 활성화합니다.
7. **[🚀 Start Training]** 버튼 클릭:
   * 학습이 진행되며 하단 로그 뷰어에 에포크별 Loss(손실)와 mIoU 성능이 실시간 기록됩니다.
   * 학습 완료 시 `checkpoints/[데이터셋명]/[모델명_시간]/best.pt` 경로에 최적 가중치가 저장됩니다.

---

## 🎯 Step 2. 추론하기 (Inference)

학습 완료된 AI 모델을 가져와 새로운 대상 이미지들에 분할 예측을 수행하는 단계입니다.

1. **[Inference]** 탭을 선택합니다.
2. **Checkpoint(s) (가중치 파일)**:
   * `[Browse]` 버튼을 눌러 Step 1에서 저장한 모델의 `best.pt` 가중치를 선택합니다.
3. **Target Images (대상 이미지 폴더)**:
   * 추론을 수행하여 분류 지도를 얻고자 하는 타겟 이미지들이 모여있는 폴더(예: `test/image`)를 선택합니다.
4. **Uncertainty Estimation Options (불확실성 설정)**:
   * **`[☑️ Enable MC Dropout]`** 옵션을 켭니다. (추론 시 드롭아웃을 켜서 모델이 여러 번 고민하도록 만드는 원리입니다.)
   * **Seeds**: 인공지능이 무작위로 고민할 시드 번호들을 쉼표로 구분하여 적습니다. (예: `1,2,3,4,5` 입력 시 총 5번의 다중 앙상블 추론 수행)
5. **[🚀 Start Inference]** 버튼 클릭:
   * 예측 분류 맵(`ensemble_mean`)과 불확실성 시각화 맵(`ensemble_std`), 그리고 평가용 데이터 파일(`uncertainty_data/*.npz`)이 `Results` 폴더 하위에 모델 및 날짜별로 생성됩니다.

---

## 📊 Step 3. 불확실성 평가하기 (Evaluate Uncertainty)

모델이 헷갈려하는 부분과 실제 오답 간의 정합성을 측정하고 논문용 이미지 패널을 자동 구성하는 단계입니다.

1. **[Inference]** 탭 하단의 주황색 **[🔥 Evaluate Uncertainty]** 버튼을 클릭합니다.
2. **결과 폴더 선택**:
   * Step 2에서 생성된 결과 폴더(예: `MASK2FORMER_Inference_20260623_144030`)를 지정합니다.
3. **[스마트 스위치 기능 작동]**:
   * 만약 선택한 결과 폴더의 모델(예: U-Net)과 현재 UI 상단에 지정된 `Model Architecture` 선택값(예: Mask2Former)이 다를 경우, 플러그인이 이를 자동으로 감지하여 경고 창을 띄웁니다:
     > *"UI 설정을 'UNET' 모델로 전환하고 계속 진행하시겠습니까?"*
   * **`[예(Yes)]`**를 누르면 자동으로 UI 드롭다운이 변경되며 평가 프로세스가 실행됩니다. (실수로 잘못된 모델명 접두사가 저장되는 것을 완벽 차단)
4. **Label Directory (정답 라벨 폴더)**:
   * 팝업창에서 성능 평가의 기준이 될 실제 정답 라벨 이미지 폴더(예: `test/label`)를 지정합니다.
5. **[자동 출력 및 사본 생성 완료]**:
   * 실행 즉시 결과 폴더에 아래의 차트 파일들이 자동 생성되며, **논문 작성 시 파일 덮어쓰기 혼선을 방지하기 위해 모델 접두사가 붙은 사본 파일도 자동 생성**됩니다.
     * `sparsification.png` ➔ `[모델명]_sparsification.png`
     * `confusion_matrix.png` ➔ `[모델명]_confusion_matrix.png`
     * `metrics_uncertainty.json` ➔ `[모델명]_metrics_uncertainty.json`
   * **Qualitative Panels (5-패널 비교 이미지)**:
     * `Qualitative_Panels/` 폴더 하위에 **10개의 5-패널 무작위 예시 이미지**(`[원본 이미지 - 정답 라벨 - 모델 예측 - 에러 지도 - 불확실성 지도]`)가 자동 배치됩니다.
     * 마찬가지로 논문 삽입이 편리하도록 `[모델명]_panel_01_[이미지명].png` 형식의 사본도 함께 자동 저장됩니다.
     * 5번째인 불확실성 지도 칸의 제목은 설정한 평가 방식에 따라 `Uncertainty (BALD)`, `Uncertainty (STD)` 등으로 동적 표시됩니다.

---

## 🏆 Step 4. 결과 분석 및 비교 (Results & Comparison)

### 📈 단일 모델 결과 확인 (Results Tab)
* **[Results]** 탭으로 이동하여 왼쪽 **History List**에서 완료된 평가 내역을 클릭합니다.
* 우측 대시보드에 종합 정확도(OA), Kappa 지표, 클래스별 IoU 및 검수 효율성이 테이블로 자동 렌더링됩니다.
* 하단의 **[📊 View HITL Curve]**, **[📉 View Sparsification]**, **[🖼️ Open Image Panels]** 버튼을 통해 고화질 이미지 리포트를 즉시 열어볼 수 있습니다.

### 🏆 다중 모델 교차 비교 (Cross-Model Comparison)
* **Compare Training Curves (학습 속도 비교)**:
  * `[Compare Training Curves]` 버튼을 눌러 비교하고 싶은 모델들의 `checkpoints` 폴더 내 다중 csv 학습 이력들을 선택하면, 에폭에 따른 Loss와 mIoU 학습 곡선이 병합된 하나의 그래프가 생성됩니다.
* **Compare Inference & Uncertainty (최종 추론/불확실성 성능 비교)**:
  * `[Compare Inference & Uncertainty]` 버튼을 눌러 비교할 다중 모델의 결과 폴더 상위 경로를 선택하면, 각 모델의 정량 지표가 비교된 **대조 대시보드 막대그래프**와 상세 엑셀 파일(`comparison_report.csv`)이 생성되어 논문 작성 시간을 대폭 단축해 줍니다.
