# QGIS_GeoActive ML Studio: QGIS Plugin User Guide

본 QGIS 플러그인은 위성/항공 영상(Remote Sensing) 및 비전 데이터의 의미론적 분할(Semantic Segmentation)을 수행하고, 학술 논문에 즉시 활용 가능한 **불확실성(Uncertainty) 평가 및 다중 베이스라인 벤치마킹**을 지원하는 강력한 연구용 도구입니다.

---

## 🌟 1. 핵심 기능 요약

1. **4대 메이저 아키텍처 지원**: `Mask2Former`, `U-Net`, `DeepLabV3+`, `SegFormer` 중 원하는 모델을 드롭다운에서 쉽게 선택하여 비교 실험을 수행할 수 있습니다.
2. **다중 시드(Multi-Seed) 앙상블 및 MC Dropout**: 신뢰성 높은 논문용 데이터(Mean ± SD) 추출을 위해 한 번의 버튼 클릭으로 여러 개의 시드를 자동 평가합니다.
3. **4대 불확실성 기법 전수 평가**: 예측 과정에서 `BALD(Epistemic)`, `Entropy`, `Max-Softmax`, `STD` 기반의 불확실성을 동시 산출하여 수치화(AUROC, AUSE)합니다.
4. **Human-in-the-Loop (HITL) 시뮬레이션**: 불확실성이 높은 지역(타일 단위)만을 우선적으로 검수할 때의 효율(검수 예산 대비 오류 검출률)을 자동으로 계산하여 논문 방어용 데이터를 제공합니다.
5. **QGIS 대시보드 (HTML Table)**: 모든 정량적 결과(Confusion Matrix, Class별 IoU, 검수 효율 곡선 등)가 QGIS 우측 패널의 HTML 표 형태로 아름답게 렌더링됩니다.

---

## 🛠️ 2. 단계별 사용 가이드

### Step 1. 글로벌 환경 설정 (1. Environment Settings)
* **Color / DN Mapping**: 플러그인 하단의 `Add Class` 버튼을 눌러 모델이 학습할 클래스(배경, 객체1, 객체2 등)의 색상과 픽셀값(DN)을 매핑합니다. 
* **16-bit Tiff**: 사용하시는 위성 영상이 16비트일 경우 해당 옵션을 체크합니다.
* **Model Architecture**: 학습 및 추론에 사용할 딥러닝 아키텍처를 선택합니다. (비교 논문 작성 시 유용합니다)

### Step 2. 모델 학습 (2. Train Model)
* **Dataset Directory**: `image` 폴더와 `label` 폴더가 포함된 최상위 데이터셋 경로를 지정합니다.
* **Epochs & Batch Size**: GPU 메모리에 맞게 하이퍼파라미터를 조절합니다.
* **Start Training**: 버튼을 누르면 `checkpoints/C_데이터셋이름/모델이름_데이터셋이름_시간/` 폴더에 체크포인트가 저장됩니다.

### Step 3. 추론 및 불확실성 평가 (3. Inference & Uncertainty)
* **Checkpoint File**: Step 2에서 학습된 `.pth` 체크포인트 파일을 다중 선택(콤마 `,` 로 구분)하여 앙상블 추론을 진행할 수 있습니다.
* **Images Directory**: 추론하고자 하는 대상 이미지가 있는 폴더를 지정합니다.
* **Seeds & MC Dropout**: 시드 번호를 콤마로 구분하여 입력하고(예: `42, 43, 44, 45, 46`), **MC Dropout**을 체크하면 하나의 체크포인트만으로도 불확실성 지도를 생성할 수 있습니다.
* **Start Inference**: 추론이 시작되면 예측 이미지와 불확실성 데이터(.npz)가 `Results/R_데이터셋이름/모델이름_InferenceResults/` 폴더에 저장됩니다.

### Step 4. 결과 분석 및 논문 작성 (4. Evaluation History)
* 추론이 끝나면 플러그인 우측 **History List**에 완료 내역이 추가됩니다.
* 내역을 클릭하면 화면에 HTML 대시보드가 열리며, `mIoU`, `OA`, `AUROC` 지표가 **"평균 ± 표준편차"** 형태로 제공됩니다.
* 해당 폴더 안의 `metrics_uncertainty.json` 파일을 열면 **Methodology Details** 항목이 들어있습니다. 논문의 Implementation Details 섹션에 해당 문구를 그대로 복사해 붙여넣으시면 됩니다.

---

## 💡 3. 논문 작성을 위한 꿀팁 (Methodology Details)
평가 결과 콘솔이나 `metrics_uncertainty.json` 내부에는 논문 작성을 돕기 위한 방법론 세부 정보가 기재되어 있습니다. 

**[Copy & Paste 예시]**
> (i) Dropout rate: 0.1 (MC Dropout 및 Stochastic Depth 적용)
> (ii) Inference Background Threshold: 0.0 (기본값 0.0으로 비활성화하여 순수 Argmax 예측 수행)
> (iii) ECE (Expected Calibration Error) 설정: 15 bins (0.0 ~ 1.0 구간 등간격 분할)
> (iv) HITL (Human-in-the-Loop) 기준: 512x512 타일 단위 분할, 타일 내 유효 픽셀 불확실성의 '평균(Mean)' 산출

이 문구들을 활용하여 Reviewer들에게 체계적인 실험 환경을 어필할 수 있습니다.

---

## ⚠️ 4. 주의사항 (Troubleshooting)
* **메모리 부족 (OOM)**: `Batch Size`를 줄이거나, `Inference` 시 이미지 크기가 너무 크다면 이미지를 분할하여 넣어주세요.
---

## 📚 5. 탑재된 모델 아키텍처 상세 제원 (Model Specifications)
학술 논문에 모델의 구조적 특징과 사전 학습(Pre-trained) 가중치 정보를 기재하실 때 아래의 상세 제원을 참고하시기 바랍니다.

*   **Mask2Former (Masked-attention Mask Transformer)**
    *   **Backbone (인코더)**: Swin-Large (또는 Swin-Tiny)
    *   **특징**: 픽셀 수준이 아닌 마스크 쿼리(Mask Query) 기반의 최신 범용 세그멘테이션 모델입니다. 복잡한 형태나 작은 객체 분할에 압도적인 성능을 보입니다.
    *   **Pre-trained**: Cityscapes 및 ImageNet 가중치 기반
*   **U-Net**
    *   **Backbone (인코더)**: ResNet-50 (`segmentation-models-pytorch` 활용)
    *   **특징**: 의료 영상 및 원격탐사 분야의 가장 고전적이고 안정적인 베이스라인입니다. 깊은 ResNet-50 인코더를 사용하여 일반 U-Net보다 뛰어난 특징 추출 성능을 가집니다.
    *   **Pre-trained**: ImageNet 가중치 기반
*   **DeepLabV3+**
    *   **Backbone (인코더)**: ResNet-50 (`segmentation-models-pytorch` 활용)
    *   **특징**: ASPP (Atrous Spatial Pyramid Pooling) 모듈을 사용하여 다중 스케일(Multi-scale)의 문맥 정보를 획득하는 데 유리합니다.
    *   **Pre-trained**: ImageNet 가중치 기반
*   **SegFormer**
    *   **Backbone (인코더)**: MiT-b2 (Mix Transformer encoder)
    *   **특징**: 가볍고 강력한 트랜스포머 기반 모델로, 위치 인코딩(Positional Encoding)이 필요 없는 계층적 트랜스포머 구조를 사용하여 위성 영상과 같이 해상도가 큰 이미지 처리에 매우 효율적입니다.
    *   **Pre-trained**: ImageNet 가중치 기반 (`nvidia/mit-b2`)
