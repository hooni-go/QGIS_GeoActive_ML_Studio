# -*- coding: utf-8 -*-
"""
Main UI definition for the Advanced Remote Sensing Mask2Former Plugin.
Supports Custom Python Environment, 3-band/4-band selection, and Train/Inference modes.
"""

import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QFileDialog, QSpinBox, 
                             QDoubleSpinBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QProgressBar, QTextEdit, QMessageBox, QApplication,
                             QComboBox, QTabWidget, QWidget, QCheckBox, QGroupBox, QSplitter, QListWidget)
from PyQt5.QtCore import Qt

class AdvancedRSDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("QGIS_GeoActive ML Studio")
        self.resize(800, 800)
        
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. Environment Setup Panel
        env_group = QVBoxLayout()
        env_group.addWidget(QLabel("<b>1. Environment Settings</b>"))
        
        env_layout = QHBoxLayout()
        self.python_env_input = QLineEdit()
        self.python_env_input.setPlaceholderText("Auto-detected python_env...")
        self.python_env_input.setReadOnly(True)
        self.python_env_input.setStyleSheet("background-color: #e0e0e0; color: #555;")
        
        env_layout.addWidget(QLabel("Portable Env Dir:"))
        env_layout.addWidget(self.python_env_input)
        
        # Model Architecture Combo (Global)
        env_layout.addSpacing(20)
        env_layout.addWidget(QLabel("<b>Model Architecture:</b>"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Mask2Former (Swin-L)", "U-Net (ResNet50)", "DeepLabV3+ (ResNet50)", "SegFormer (MiT-B2)"])
        self.model_combo.setMinimumWidth(180)
        env_layout.addWidget(self.model_combo)
        
        env_group.addLayout(env_layout)
        
        main_layout.addLayout(env_group)
        main_layout.addSpacing(10)

        # Tabs for Train / Inference / Info / Results / Active Learning
        self.tabs = QTabWidget()
        self.train_tab = QWidget()
        self.infer_tab = QWidget()
        self.info_tab = QWidget()
        self.results_tab = QWidget()
        self.active_tab = QWidget()
        self.tabs.addTab(self.train_tab, "Training")
        self.tabs.addTab(self.infer_tab, "Inference")
        self.tabs.addTab(self.info_tab, "Model Info & Guide")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.active_tab, "Active Learning")
        main_layout.addWidget(self.tabs)

        self.setup_train_tab()
        self.setup_infer_tab()
        self.setup_info_tab()
        self.setup_results_tab()
        self.setup_active_tab()

        # 4. Progress and Log
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        main_layout.addWidget(self.log_text)

    def setup_train_tab(self):
        layout = QVBoxLayout(self.train_tab)
        
        # 1. Dataset Configuration Group
        data_group = QGroupBox("1. Dataset Configuration")
        data_group_layout = QVBoxLayout()
        
        guide_dataset = QLabel("💡 가이드: Dataset Root는 하위에 train/image, train/label, val/image, val/label 구조를 지녀야 합니다.")
        guide_dataset.setStyleSheet("color: #006699; font-size: 11px; font-style: italic; margin-bottom: 4px;")
        data_group_layout.addWidget(guide_dataset)
        
        data_layout = QHBoxLayout()
        self.dataset_dir_input = QLineEdit()
        self.dataset_dir_input.setPlaceholderText("Dataset Root (contains train/val/test)...")
        self.dataset_dir_input.setToolTip("학습 데이터셋의 최상위 루트 폴더입니다. (하위에 train 및 val 폴더 필수)")
        data_btn = QPushButton("Browse")
        data_btn.clicked.connect(lambda: self.browse_folder(self.dataset_dir_input))
        data_layout.addWidget(QLabel("Dataset Root:"))
        data_layout.addWidget(self.dataset_dir_input)
        data_layout.addWidget(data_btn)
        data_group_layout.addLayout(data_layout)
        
        type_layout = QHBoxLayout()
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems(["Aerial 3-Band (RGB, 8-bit)", "Satellite 4-Band (RGBN, 16-bit)"])
        self.data_type_combo.setToolTip("입력 데이터의 형식과 밴드 구성을 지정합니다. 위성 영상은 RGB-NearInfrared 4밴드 로더를 사용합니다.")
        type_layout.addWidget(QLabel("Data Type:"))
        type_layout.addWidget(self.data_type_combo)
        data_group_layout.addLayout(type_layout)
        
        data_group.setLayout(data_group_layout)
        layout.addWidget(data_group)
        
        # 2. Training Hyperparameters Group
        hyper_group = QGroupBox("2. Training Hyperparameters")
        hyper_layout = QVBoxLayout()
        
        guide_hyper = QLabel("💡 가이드: Epoch은 100 이상을 권장하며, Focal Loss 활성화 시 클래스 불균형에 가중치를 부여해 경계 오류를 억제합니다.")
        guide_hyper.setStyleSheet("color: #006699; font-size: 11px; font-style: italic; margin-bottom: 4px;")
        hyper_layout.addWidget(guide_hyper)
        
        hyper_row1 = QHBoxLayout()
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        self.epochs_spin.setToolTip("모델 학습의 전체 반복 횟수(Epoch)를 설정합니다. 기본 권장 값은 100입니다.")
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(4)
        self.batch_spin.setToolTip("배치 크기(Batch Size)입니다. GPU 성능(VRAM 용량)에 따라 4 혹은 8 이상으로 설정 가능합니다.")
        
        hyper_row1.addWidget(QLabel("Epochs:"))
        hyper_row1.addWidget(self.epochs_spin)
        hyper_row1.addWidget(QLabel("Batch Size:"))
        hyper_row1.addWidget(self.batch_spin)
        
        focal_layout = QHBoxLayout()
        self.focal_loss_cb = QCheckBox("Enable Focal Loss")
        self.focal_loss_cb.setChecked(False)
        self.focal_loss_cb.setToolTip("Focal Loss를 적용하여 판별이 까다로운 탐지 경계 및 미세 건물 픽셀에 동적인 학습 패널티 가중치를 부여합니다.")
        
        self.focal_gamma_spin = QDoubleSpinBox()
        self.focal_gamma_spin.setRange(0.1, 5.0)
        self.focal_gamma_spin.setValue(2.0)
        self.focal_gamma_spin.setSingleStep(0.5)
        self.focal_gamma_spin.setEnabled(False)
        self.focal_gamma_spin.setToolTip("Gamma (Focal Loss 감쇠 계수. 2.0 권장)")
        
        self.focal_alpha_spin = QDoubleSpinBox()
        self.focal_alpha_spin.setRange(0.1, 20.0)
        self.focal_alpha_spin.setValue(5.0)
        self.focal_alpha_spin.setSingleStep(1.0)
        self.focal_alpha_spin.setEnabled(False)
        self.focal_alpha_spin.setToolTip("Alpha (보조 난이도 제어 가중치. 5.0 권장)")
        
        self.focal_loss_cb.toggled.connect(self.focal_gamma_spin.setEnabled)
        self.focal_loss_cb.toggled.connect(self.focal_alpha_spin.setEnabled)
        
        focal_layout.addWidget(self.focal_loss_cb)
        focal_layout.addWidget(QLabel("γ(Gamma):"))
        focal_layout.addWidget(self.focal_gamma_spin)
        focal_layout.addWidget(QLabel("α(Weight):"))
        focal_layout.addWidget(self.focal_alpha_spin)
        
        hyper_row1.addLayout(focal_layout)
        hyper_layout.addLayout(hyper_row1)
        
        # Class Weights Table
        self.weights_table = QTableWidget(0, 3)
        self.weights_table.setHorizontalHeaderLabels(["Model ID", "Class Name", "Loss Weight"])
        self.weights_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.weights_table.setFixedHeight(150)
        self.weights_table.setVisible(False)
        self.weights_table.setToolTip("각 클래스별 학습 가중치입니다. 클래스별 면적 편차에 따른 편향을 교정합니다.")
        self.focal_loss_cb.toggled.connect(self.weights_table.setVisible)
        hyper_layout.addWidget(self.weights_table)
        
        self.btn_auto_weight = QPushButton("🪄 Auto-Calculate Optimal Weights")
        self.btn_auto_weight.setStyleSheet("background-color: #03A9F4; color: white; font-weight: bold; border-radius: 4px; padding: 5px;")
        self.btn_auto_weight.setVisible(False)
        self.btn_auto_weight.setToolTip("학습 셋 내 클래스 점유 비율을 자동 역산하여, 최적의 손실 함수 가중치(Class Weight)를 산출해 테이블에 채웁니다.")
        self.focal_loss_cb.toggled.connect(self.btn_auto_weight.setVisible)
        hyper_layout.addWidget(self.btn_auto_weight)
        
        hyper_group.setLayout(hyper_layout)
        layout.addWidget(hyper_group)
        
        # 3. Transfer Learning Group
        tl_group = QGroupBox("3. Transfer Learning (Optional)")
        tl_layout = QVBoxLayout()
        
        guide_tl = QLabel("💡 가이드: 기존 학습 중이던 best.pt를 선택하여 이어서 Fine-tuning을 진행할 수 있습니다. (미선택 시 신규 학습)")
        guide_tl.setStyleSheet("color: #006699; font-size: 11px; font-style: italic; margin-bottom: 4px;")
        tl_layout.addWidget(guide_tl)
        
        tl_row = QHBoxLayout()
        self.train_ckpt_input = QLineEdit()
        self.train_ckpt_input.setPlaceholderText("Pre-trained Checkpoint (.pt) ...")
        self.train_ckpt_input.setToolTip("미세 조정(Fine-tuning)을 위한 기학습 가중치 파일(.pt) 경로입니다.")
        train_ckpt_btn = QPushButton("Browse")
        train_ckpt_btn.clicked.connect(self.browse_transfer_checkpoint)
        tl_row.addWidget(QLabel("Pre-trained Checkpoint:"))
        tl_row.addWidget(self.train_ckpt_input)
        tl_row.addWidget(train_ckpt_btn)
        tl_layout.addLayout(tl_row)
        
        tl_group.setLayout(tl_layout)
        layout.addWidget(tl_group)
        
        # Actions
        self.train_btn = QPushButton("Start Training")
        self.train_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; height: 40px;")
        self.train_btn.setToolTip("설정된 파라미터로 모델 학습 프로세스를 백그라운드로 기동합니다.")
        
        self.view_config_btn = QPushButton("View Training Config")
        self.view_config_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; height: 40px;")
        self.view_config_btn.setToolTip("현재 저장된 최신의 학습 json 설정 파일을 조회합니다.")
        
        self.view_metrics_btn = QPushButton("View Training Metrics")
        self.view_metrics_btn.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; height: 40px;")
        self.view_metrics_btn.setToolTip("학습 에폭별 Loss 및 mIoU 변화 로그 표를 조회합니다.")
        
        self.view_graphs_btn = QPushButton("View Training Graphs")
        self.view_graphs_btn.setStyleSheet("background-color: #E91E63; color: white; font-weight: bold; height: 40px;")
        self.view_graphs_btn.setToolTip("학습 수렴 추세(Loss/Validation mIoU 곡선) 시각화 그래프 이미지를 표시합니다.")
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.train_btn)
        btn_layout.addWidget(self.view_config_btn)
        btn_layout.addWidget(self.view_metrics_btn)
        btn_layout.addWidget(self.view_graphs_btn)
        layout.addLayout(btn_layout)
        
        layout.addStretch()

    def setup_infer_tab(self):
        layout = QVBoxLayout(self.infer_tab)
        
        # 1. Checkpoint Selection
        ckpt_group = QGroupBox("1. Model Configuration")
        ckpt_layout = QVBoxLayout()
        
        guide_ckpt = QLabel("💡 가이드: 학습 완료된 .pt 파일을 선택하세요. 다중 선택 시 심층 앙상블(Deep Ensemble)이 자동 기동됩니다.")
        guide_ckpt.setStyleSheet("color: #006699; font-size: 11px; font-style: italic; margin-bottom: 4px;")
        ckpt_layout.addWidget(guide_ckpt)
        
        ckpt_row = QHBoxLayout()
        self.ckpt_input = QLineEdit()
        self.ckpt_input.setPlaceholderText("Select Model Checkpoint(s) (.pt)... (Deep Ensemble: select multiple)")
        self.ckpt_input.setToolTip("추론에 사용할 학습된 모델 가중치 파일(.pt) 경로입니다. Ctrl을 눌러 여러 가중치 파일을 복수 지정하면 앙상블 평균 추론을 수행합니다.")
        ckpt_btn = QPushButton("Browse")
        ckpt_btn.clicked.connect(lambda: self.browse_files(self.ckpt_input, "PyTorch Model (*.pt *.pth)"))
        ckpt_row.addWidget(QLabel("Checkpoint(s):"))
        ckpt_row.addWidget(self.ckpt_input)
        ckpt_row.addWidget(ckpt_btn)
        ckpt_layout.addLayout(ckpt_row)
        
        ckpt_group.setLayout(ckpt_layout)
        layout.addWidget(ckpt_group)
        
        # 2. Data Selection
        data_group = QGroupBox("2. Target Images")
        img_layout = QVBoxLayout()
        
        guide_img = QLabel("💡 가이드: 탐지(추론)를 수행할 원본 TIF 항공/위성 영상들이 담긴 폴더를 지정합니다.")
        guide_img.setStyleSheet("color: #006699; font-size: 11px; font-style: italic; margin-bottom: 4px;")
        img_layout.addWidget(guide_img)
        
        img_row = QHBoxLayout()
        self.infer_img_input = QLineEdit()
        self.infer_img_input.setPlaceholderText("Select Target Images Directory...")
        self.infer_img_input.setToolTip("추론 대상 영상(.tif)들이 들어 있는 최상위 폴더 경로입니다.")
        img_btn = QPushButton("Browse")
        img_btn.clicked.connect(lambda: self.browse_folder(self.infer_img_input))
        img_row.addWidget(QLabel("Images Dir:"))
        img_row.addWidget(self.infer_img_input)
        img_row.addWidget(img_btn)
        img_layout.addLayout(img_row)
        
        data_group.setLayout(img_layout)
        layout.addWidget(data_group)
        
        # 3. Uncertainty Options
        unc_group = QGroupBox("3. Uncertainty Estimation Options")
        unc_layout = QVBoxLayout()
        
        guide_unc = QLabel("💡 가이드: MC Dropout 체크 후 20개 시드를 지정하면, 픽셀의 확률 분포 분석을 통해 신뢰성이 가미된 불확실성 지도가 동시 빌드됩니다.")
        guide_unc.setStyleSheet("color: #006699; font-size: 11px; font-style: italic; margin-bottom: 4px;")
        unc_layout.addWidget(guide_unc)
        
        seed_layout = QHBoxLayout()
        self.seeds_input = QLineEdit()
        self.seeds_input.setText("1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
        self.seeds_input.setPlaceholderText("Comma-separated seeds...")
        self.seeds_input.setToolTip("추론 시드 조합입니다. 다중 시드 추론은 앙상블 효과를 주며 20개 이상의 시드 입력을 권장합니다.")
        self.mc_dropout_cb = QCheckBox("Enable MC Dropout")
        self.mc_dropout_cb.setToolTip("추론 단계에서도 드롭아웃 레이어를 활성화하여 데이터의 내재적 확률 분포를 추정하는 베이지안 확률 추론을 가동합니다.")
        seed_layout.addWidget(QLabel("Seeds:"))
        seed_layout.addWidget(self.seeds_input)
        seed_layout.addWidget(self.mc_dropout_cb)
        unc_layout.addLayout(seed_layout)
        
        unc_group.setLayout(unc_layout)
        layout.addWidget(unc_group)
        
        # Actions Layout
        action_layout = QHBoxLayout()
        self.infer_btn = QPushButton("Start Inference")
        self.infer_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; height: 40px;")
        self.infer_btn.setToolTip("설정된 가중치와 시드를 통해 백그라운드 추론 연산 프로세스를 시작합니다.")
        
        self.eval_uncertainty_btn = QPushButton("Evaluate Uncertainty")
        self.eval_uncertainty_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; height: 40px;")
        self.eval_uncertainty_btn.setToolTip("추론된 불확실성 지도(.npz) 데이터를 기반으로 에러 탐지율, 오차 검증 요약 통계(AUROC, AUSE, ECE)를 자동 분석해 결과를 도출합니다.")
        
        action_layout.addWidget(self.infer_btn)
        action_layout.addWidget(self.eval_uncertainty_btn)
        layout.addLayout(action_layout)
        layout.addStretch()

    def setup_info_tab(self):
        layout = QVBoxLayout(self.info_tab)
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        
        html_content = """
        <h2 style='color: #2E86C1;'>플러그인 작업 흐름 및 가이드 (Plugin Workflow)</h2>
        <h3>1. 데이터 준비 및 모델 학습 (Training)</h3>
        <ul>
            <li><b>Environment Settings</b>에서 환경 설정 및 타겟 <b>Model Architecture</b>를 선택합니다.</li>
            <li>학습용 이미지가 <code>train/image</code> 및 <code>train/label</code> 폴더에 올바르게 분류되어 있는지 확인합니다.</li>
            <li><b>Transfer Learning (선택):</b> <b>Training</b> 탭에서 기존에 학습된 <code>.pt</code> 가중치 파일을 선택하여 이어서 학습(Fine-tuning)을 수행할 수 있습니다. 비워두면 ImageNet 기본 가중치로 시작합니다.</li>
        </ul>
        <h3>2. 불확실성 평가 및 추론 (Inference)</h3>
        <ul>
            <li><b>Inference</b> 탭으로 이동합니다.</li>
            <li>쉼표로 구분된 <b>Seeds</b> 번호를 입력합니다 (예: 42,43,44,45,46).</li>
            <li>추론 과정에 확률적 변동성을 주어 불확실성을 산출하려면 <b>Enable MC Dropout</b>을 체크합니다.</li>
            <li><b>Start Inference</b> 버튼을 클릭합니다. 모델이 설정한 시드 개수만큼 다중 추론을 수행합니다.</li>
        </ul>
        <h3>3. 결과 분석 및 시각화 (Analysis)</h3>
        <ul>
            <li>추론이 완료되면 <b>앙상블 평균 (Ensemble Mean)</b> 결과와 <b>불확실성 지도 (Uncertainty Map)</b>가 QGIS 맵 캔버스에 자동 로드됩니다.</li>
            <li>우측의 <b>Results</b> 탭에서 논문에 바로 인용할 수 있는 실험 평가 요약 표(mIoU, ECE, HITL 등)를 확인하세요.</li>
        </ul>
        
        <hr/>
        <h2 style='color: #2E86C1;'>지원되는 모델 아키텍처 상세 (Model Architectures)</h2>
        <h3>1. Mask2Former (Masked-attention Mask Transformer)</h3>
        <ul>
            <li><b>Backbone:</b> Swin-Large (또는 Swin-Tiny)</li>
            <li><b>Feature:</b> 픽셀 단위 분류가 아닌 마스크 쿼리(Mask Query) 기반의 최신 범용 세그멘테이션 모델입니다. 복잡한 형태나 작은 객체 분할에 압도적인 성능을 보입니다.</li>
        </ul>
        <h3>2. U-Net (ResNet-50)</h3>
        <ul>
            <li><b>Backbone:</b> ResNet-50 Encoder (SMP 라이브러리 활용)</li>
            <li><b>Feature:</b> 원격탐사 및 의료 영상 분야에서 가장 신뢰받고 널리 쓰이는 안정적인 베이스라인입니다. 깊은 ResNet-50 인코더 구조를 채택했습니다.</li>
        </ul>
        <h3>3. DeepLabV3+</h3>
        <ul>
            <li><b>Backbone:</b> ResNet-50 Encoder (SMP 라이브러리 활용)</li>
            <li><b>Feature:</b> ASPP (Atrous Spatial Pyramid Pooling) 모듈을 사용하여 다중 스케일(Multi-scale)의 문맥(Context) 정보를 매우 효율적으로 포착합니다.</li>
        </ul>
        <h3>4. SegFormer</h3>
        <ul>
            <li><b>Backbone:</b> MiT-b2 (Mix Transformer encoder)</li>
            <li><b>Feature:</b> 위치 인코딩(Positional Encoding)이 필요 없는 가볍고 강력한 계층형 트랜스포머 모델로, 고해상도 위성 영상 처리에 매우 최적화되어 있습니다.</li>
        </ul>
        
        <hr/>
        <h2 style='color: #2E86C1;'>주요 핵심 기술 (Advanced AI Features)</h2>
        <h3>1. 심층 앙상블 및 불확실성 산출 (Deep Ensembles)</h3>
        <ul>
            <li>서로 다른 시드와 MC Dropout을 병합한 다중 추론을 수행하여, 인공지능이 스스로 확신하지 못하는 <b>인지적 불확실성(Epistemic Uncertainty)</b>을 포착합니다.</li>
            <li>이 과정은 예측 오류 가능성이 높은 지역을 붉게 강조하는 표준편차 지도를 생성해 냅니다.</li>
        </ul>
        <h3>2. 보조 Focal Loss (동적 난이도 조절)</h3>
        <ul>
            <li>커스텀 픽셀 단위 Focal Loss가 오분류된 '어려운 픽셀(얇은 도로 등)'에 지수적으로 패널티를 부여합니다.</li>
            <li>이는 모델이 쉬운 영역의 정확도만 높이려는 꼼수를 막고, 자신의 약점에 집중하여 세밀하게 학습하도록 강제합니다.</li>
        </ul>
        <h3>3. Human-in-the-Loop (HITL) 기반 검수 자동화</h3>
        <ul>
            <li>산출된 불확실성 지표를 통해 수동 육안 검수 비용을 획기적으로 줄일 수 있습니다.</li>
            <li>전체 지도의 100%를 무식하게 검수할 필요 없이, 플러그인이 안내하는 불확실성 상위 20% 타일 영역만 우선 검수하면 AI 오류의 대다수(7~80% 이상)를 단번에 교정할 수 있습니다.</li>
        </ul>
        <h3>4. 자동 혼합 정밀도 (AMP) 및 연산 안정성</h3>
        <ul>
            <li>학습 속도 향상과 GPU VRAM 효율성을 극대화하기 위해 <code>float16</code> 연산을 적극 활용합니다.</li>
            <li>동시에, 손실 함수 계산 시 NaN 발산 폭발을 막기 위한 강제 Clamping 및 <code>float32</code> 안전 캐스팅이 적용되어 있습니다.</li>
        </ul>
        """
        info_text.setHtml(html_content)
        layout.addWidget(info_text)

    def setup_results_tab(self):
        layout = QVBoxLayout(self.results_tab)
        
        # --- NEW: Cross-Model Comparison Group ---
        comp_group = QGroupBox("🏆 Cross-Model Comparison")
        comp_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid silver; border-radius: 6px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        comp_layout = QHBoxLayout(comp_group)
        
        self.btn_compare_train = QPushButton("📈 Compare Training Curves")
        self.btn_compare_train.setStyleSheet("background-color: #E8F8F5; font-weight: bold; padding: 8px;")
        
        self.btn_compare_infer = QPushButton("📊 Compare Inference & Uncertainty")
        self.btn_compare_infer.setStyleSheet("background-color: #FEF9E7; font-weight: bold; padding: 8px;")
        
        comp_layout.addWidget(self.btn_compare_train)
        comp_layout.addWidget(self.btn_compare_infer)
        layout.addWidget(comp_group)
        # ----------------------------------------
        
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel: History List + Load Button
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_load_res = QPushButton("📂 Load Past Result Folder")
        self.btn_load_res.setStyleSheet("background-color: #EBF5FB; font-weight: bold; padding: 6px; border: 1px solid #AED6F1; border-radius: 4px;")
        self.btn_load_res.setToolTip("이전에 평가 완료된 결과 폴더를 불러와 대시보드 및 그래프 분석 기능을 다시 활성화합니다.")
        
        self.history_list = QListWidget()
        self.history_list.setToolTip("현재 세션 및 로드된 과거 평가 리스트입니다. 항목 클릭 시 대시보드가 로드됩니다.")
        
        left_layout.addWidget(self.btn_load_res)
        left_layout.addWidget(self.history_list)
        
        self.splitter.addWidget(left_widget)
        
        # Right Panel: Analytics Dashboard
        dash_widget = QWidget()
        dash_layout = QVBoxLayout(dash_widget)
        
        dash_title = QLabel("<b>Result Analytics Dashboard</b>")
        dash_title.setStyleSheet("font-size: 16px; color: #2E86C1;")
        dash_layout.addWidget(dash_title)
        
        self.dash_content = QTextEdit()
        self.dash_content.setReadOnly(True)
        self.dash_content.setHtml("<p style='color: gray;'>Select a run from the history list to view analytics.</p>")
        dash_layout.addWidget(self.dash_content)
        
        # Buttons for Graphs
        graph_btn_layout = QHBoxLayout()
        self.btn_hitl = QPushButton("📊 View HITL Curve")
        self.btn_sparsification = QPushButton("📉 View Sparsification")
        self.btn_cm = QPushButton("🔥 View Confusion Matrix")
        self.btn_panels = QPushButton("🖼️ Open Image Panels")
        
        for btn in [self.btn_hitl, self.btn_sparsification, self.btn_cm, self.btn_panels]:
            btn.setEnabled(False)
            btn.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
            graph_btn_layout.addWidget(btn)
            
        dash_layout.addLayout(graph_btn_layout)
        self.splitter.addWidget(dash_widget)
        
        self.splitter.setSizes([250, 550])
        layout.addWidget(self.splitter)

    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            line_edit.setText(folder)
            
    def browse_file(self, line_edit, filter_str):
        file, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter_str)
        if file:
            line_edit.setText(file)
            
    def browse_transfer_checkpoint(self):
        self.browse_file(self.train_ckpt_input, "PyTorch Model (*.pt *.pth)")
        ckpt_path = self.train_ckpt_input.text().strip()
        if not ckpt_path or not os.path.isfile(ckpt_path):
            return
            
        import json
        ckpt_dir = os.path.dirname(ckpt_path)
        config_path = os.path.join(ckpt_dir, "train_config.json")
        
        if os.path.isfile(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                if "batch_size" in config:
                    self.batch_spin.setValue(int(config["batch_size"]))
                if "use_focal_loss" in config:
                    use_focal = str(config["use_focal_loss"]).lower() == "true"
                    self.focal_loss_cb.setChecked(use_focal)
                    if use_focal:
                        if "focal_gamma" in config and isinstance(config["focal_gamma"], (int, float)):
                            self.focal_gamma_spin.setValue(float(config["focal_gamma"]))
                        if "focal_alpha" in config and isinstance(config["focal_alpha"], (int, float)):
                            self.focal_alpha_spin.setValue(float(config["focal_alpha"]))
                            
                if "class_weights" in config:
                    weights_dict = config["class_weights"]
                    for row in range(self.weights_table.rowCount()):
                        model_id_item = self.weights_table.item(row, 0)
                        if model_id_item:
                            mid = model_id_item.text()
                            if mid in weights_dict:
                                spin = self.weights_table.cellWidget(row, 2)
                                if spin:
                                    spin.setValue(float(weights_dict[mid]))
                
                # Fine-tuning 추천 세팅
                self.epochs_spin.setValue(50)
                
                # HTML formatted message
                batch_val = config.get("batch_size", "N/A")
                gamma_val = config.get("focal_gamma", "N/A") if use_focal else "Disabled"
                alpha_val = config.get("focal_alpha", "N/A") if use_focal else "Disabled"
                
                msg_html = f"""
                <h3 style='color: #4CAF50;'>✅ Auto-Sync Complete</h3>
                <p>Loaded optimal settings from the selected model's configuration.</p>
                <table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%; font-size: 13px;'>
                    <tr style='background-color: #f2f2f2;'><th>Parameter</th><th>Auto-Loaded Value</th></tr>
                    <tr><td><b>Batch Size</b></td><td style='text-align: center;'>{batch_val}</td></tr>
                    <tr><td><b>Focal Loss</b></td><td style='text-align: center;'>{'<b>Enabled</b>' if use_focal else 'Disabled'}</td></tr>
                    <tr><td><b>Focal Gamma (γ)</b></td><td style='text-align: center;'>{gamma_val}</td></tr>
                    <tr><td><b>Focal Alpha (α)</b></td><td style='text-align: center;'>{alpha_val}</td></tr>
                    <tr><td><b>Epochs</b></td><td style='text-align: center;'><span style='color: #E91E63; font-weight:bold;'>50</span> (Fine-Tuning Rec.)</td></tr>
                </table>
                <p style='font-size:11px; color:gray; margin-top: 5px;'>* You can still manually adjust these values before training.</p>
                """
                
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Settings Auto-Configured")
                msg_box.setTextFormat(Qt.RichText)
                msg_box.setText(msg_html)
                msg_box.setIcon(QMessageBox.Information)
                msg_box.exec_()
            except Exception as e:
                print(f"Failed to auto-load config: {e}")
            
    def browse_files(self, line_edit, filter_str):
        files, _ = QFileDialog.getOpenFileNames(self, "Select File(s)", "", filter_str)
        if files:
            line_edit.setText(", ".join(files))
            
    def log(self, message):
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def setup_active_tab(self):
        layout = QVBoxLayout(self.active_tab)
        
        # 1. Pipeline Control Panel
        control_group = QGroupBox("1. Active Retraining Loop Configuration")
        control_layout = QVBoxLayout()
        
        guide_active = QLabel("💡 가이드: 이 탭은 불확실 영역을 찾아 라벨 노이즈를 필터링하고 검수 완료 시 재학습을 유도하는 Active Learning 제어 영역입니다.")
        guide_active.setStyleSheet("color: #006699; font-size: 11px; font-style: italic; margin-bottom: 4px;")
        control_layout.addWidget(guide_active)
        
        # Enable Checkbox
        self.active_enabled_cb = QCheckBox("자동 재학습 알림 및 루프 기동 활성화 (Enable Auto-Retraining Loop)")
        self.active_enabled_cb.setChecked(True)
        self.active_enabled_cb.setStyleSheet("font-weight: bold; color: #333;")
        self.active_enabled_cb.setToolTip("추론 후 불확실 타일 검출 시, 사용자 안내 팝업창을 띄우고 예 버튼 클릭 시 모델 재학습(Retraining) 공정을 자동 실행합니다.")
        control_layout.addWidget(self.active_enabled_cb)
        control_layout.addSpacing(10)
        
        # Dataset Configuration row (synchronized with Training tab)
        dataset_row = QHBoxLayout()
        self.active_dataset_dir_input = QLineEdit()
        self.active_dataset_dir_input.setPlaceholderText("Select Dataset Root Folder (contains train/val/test)...")
        self.active_dataset_dir_input.setToolTip("Active Learning을 적용할 데이터셋 루트입니다. (Training 탭과 실시간 양방향 연동)")
        active_dataset_btn = QPushButton("Browse")
        active_dataset_btn.clicked.connect(lambda: self.browse_folder(self.active_dataset_dir_input))
        dataset_row.addWidget(QLabel("학습 데이터셋 루트 (Dataset Root):"))
        dataset_row.addWidget(self.active_dataset_dir_input)
        dataset_row.addWidget(active_dataset_btn)
        control_layout.addLayout(dataset_row)
        control_layout.addSpacing(10)
        
        # Thresholds Row
        thresh_row = QHBoxLayout()
        
        # Uncertainty Threshold
        thresh_row.addWidget(QLabel("불확실성 임계치 (Uncertainty Threshold):"))
        self.active_threshold_spin = QDoubleSpinBox()
        self.active_threshold_spin.setRange(0.01, 1.00)
        self.active_threshold_spin.setSingleStep(0.01)
        self.active_threshold_spin.setValue(0.15)
        self.active_threshold_spin.setToolTip("불확실성 판별 임계값입니다. (0.15 권장. 이 값 이상의 표준편차를 가지는 픽셀을 에러로 판단)")
        thresh_row.addWidget(self.active_threshold_spin)
        
        thresh_row.addSpacing(20)
        
        # Area Ratio Threshold
        thresh_row.addWidget(QLabel("최소 후보 면적 비율 (Candidate Area Ratio %):"))
        self.active_ratio_spin = QSpinBox()
        self.active_ratio_spin.setRange(1, 100)
        self.active_ratio_spin.setSuffix("%")
        self.active_ratio_spin.setValue(5)
        self.active_ratio_spin.setToolTip("타일(패치) 내에서 불확실 픽셀이 차지해야 하는 최소 면적 비율입니다. (5% 권장. 작은 오차 노이즈 유입 차단)")
        thresh_row.addWidget(self.active_ratio_spin)
        
        control_layout.addLayout(thresh_row)
        control_layout.addSpacing(10)
        
        # Reference Label Folder
        ref_row = QHBoxLayout()
        self.active_ref_dir_input = QLineEdit()
        self.active_ref_dir_input.setPlaceholderText("기본값: 데이터셋 내 test/label 폴더 자동 탐색...")
        self.active_ref_dir_input.setToolTip("불확실 픽셀 영역을 교정할 때 참고할 참조(Reference) 참값 라벨 폴더입니다. 비워두면 Dataset Root 내의 test/label을 자동 로드합니다.")
        ref_btn = QPushButton("Browse")
        ref_btn.clicked.connect(lambda: self.browse_folder(self.active_ref_dir_input))
        ref_row.addWidget(QLabel("참조 정답 라벨 디렉터리 (Reference Label Dir):"))
        ref_row.addWidget(self.active_ref_dir_input)
        ref_row.addWidget(ref_btn)
        control_layout.addLayout(ref_row)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # 2. Manual Active Learning Execution Panel
        manual_group = QGroupBox("2. Manual Retraining Diagnostic (수동 불확실성 진단 및 학습 추가)")
        manual_layout = QVBoxLayout()
        
        guide_manual = QLabel("💡 가이드: 이전에 완료된 추론 결과 폴더를 직접 선택하여 강제로 불확실성 진단 및 라벨 보정 루프를 기동합니다.")
        guide_manual.setStyleSheet("color: #006699; font-size: 11px; font-style: italic; margin-bottom: 4px;")
        manual_layout.addWidget(guide_manual)
        
        folder_row = QHBoxLayout()
        self.active_infer_folder_input = QLineEdit()
        self.active_infer_folder_input.setPlaceholderText("추론 결과 폴더 (Results/.../UNET_Inference_...) 선택...")
        self.active_infer_folder_input.setToolTip("진단할 추론 결과 디렉토리(가장 최근 수행된 Inference 폴더)입니다.")
        folder_btn = QPushButton("Browse")
        folder_btn.clicked.connect(lambda: self.browse_folder(self.active_infer_folder_input))
        folder_row.addWidget(QLabel("대상 추론 결과 폴더:"))
        folder_row.addWidget(self.active_infer_folder_input)
        folder_row.addWidget(folder_btn)
        manual_layout.addLayout(folder_row)
        manual_layout.addSpacing(10)
        
        self.active_run_diag_btn = QPushButton("🔍 수동 진단 및 라벨 보정 실행 (Run Active Retraining Diagnostic)")
        self.active_run_diag_btn.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; height: 40px;")
        self.active_run_diag_btn.setToolTip("선택한 결과 폴더 내의 .npz 데이터와 원본 이미지를 불러와 불확실 패치를 추출하고 검수 라벨을 생성해 학습 셋으로 복사합니다.")
        manual_layout.addWidget(self.active_run_diag_btn)
        
        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group)
        
        layout.addStretch()
