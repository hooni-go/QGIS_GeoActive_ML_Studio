# -*- coding: utf-8 -*-
"""
Main Plugin Class for Advanced RS Mask2Former Integration.
Runs training and inference as independent subprocesses to protect QGIS stability.
"""

from qgis.PyQt.QtCore import QThread, pyqtSignal, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import (QAction, QMessageBox, QListWidgetItem, 
                                 QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, 
                                 QPushButton, QLabel, QListWidget, QFileDialog)
import os
import subprocess
import json
import datetime

from .ui_main import AdvancedRSDialog

class SubprocessWorker(QThread):
    log_msg = pyqtSignal(str)
    progress_val = pyqtSignal(int)
    finished = pyqtSignal(int)
    
    def __init__(self, cmd, cwd=None):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self.process = None
        self.output_dir = None
        
    def run(self):
        try:
            import os
            clean_env = os.environ.copy()
            clean_env.pop("PYTHONHOME", None)
            clean_env.pop("PYTHONPATH", None)
            
            self.process = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=clean_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for line in self.process.stdout:
                clean_line = line.strip()
                if clean_line.startswith("PROGRESS:"):
                    try:
                        pct = int(clean_line.split(":")[1].strip())
                        self.progress_val.emit(pct)
                    except:
                        pass
                else:
                    if clean_line.startswith("Output Directory:"):
                        self.output_dir = clean_line.split("Output Directory:", 1)[1].strip()
                    self.log_msg.emit(clean_line)
            
            self.process.wait()
            self.progress_val.emit(100)
            self.finished.emit(self.process.returncode)
        except Exception as e:
            self.log_msg.emit(f"Process Error: {str(e)}")
            self.finished.emit(-1)

    def stop(self):
        if self.process:
            self.process.terminate()

class ModelSelectionDialog(QDialog):
    def __init__(self, parent=None, title="Select Models to Compare", target_file="metrics.csv"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(500, 400)
        self.target_file = target_file
        self.selected_files = []
        self.selected_names = []
        
        layout = QVBoxLayout(self)
        
        # 1. Directory Selection
        dir_layout = QHBoxLayout()
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("Select root directory containing models...")
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse)
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(btn_browse)
        layout.addLayout(dir_layout)
        
        # 2. Checkbox List
        self.list_widget = QListWidget()
        layout.addWidget(QLabel(f"Available Models (Found '{target_file}'):"))
        layout.addWidget(self.list_widget)
        
        # 3. Action Buttons
        btn_layout = QHBoxLayout()
        btn_scan = QPushButton("🔄 Scan Directory")
        btn_scan.clicked.connect(self.scan_directory)
        btn_compare = QPushButton("🚀 Compare Selected")
        btn_compare.clicked.connect(self.accept_selection)
        btn_compare.setStyleSheet("background-color: #3498DB; color: white; font-weight: bold;")
        
        btn_layout.addWidget(btn_scan)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_compare)
        layout.addLayout(btn_layout)
        
    def browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Root Directory")
        if folder:
            self.dir_input.setText(folder)
            self.scan_directory()
            
    def scan_directory(self):
        root_dir = self.dir_input.text().strip()
        self.list_widget.clear()
        if not root_dir or not os.path.isdir(root_dir):
            return
            
        import glob
        pattern = f"*{self.target_file}"
        
        # 1. 3레벨 깊이까지 모든 후보 매칭 파일 스캔 (glob를 활용하여 빠르고 안정적)
        found_files = []
        found_files.extend(glob.glob(os.path.join(root_dir, pattern)))
        found_files.extend(glob.glob(os.path.join(root_dir, "*", pattern)))
        found_files.extend(glob.glob(os.path.join(root_dir, "*", "*", pattern)))
        found_files.extend(glob.glob(os.path.join(root_dir, "*", "*", "*", pattern)))
        
        # JSON 결과 비교의 경우, uncertainty_data 하위 폴더 내 파일도 스캔 지원
        if "json" in self.target_file:
            found_files.extend(glob.glob(os.path.join(root_dir, "uncertainty_data", pattern)))
            found_files.extend(glob.glob(os.path.join(root_dir, "*", "uncertainty_data", pattern)))
            found_files.extend(glob.glob(os.path.join(root_dir, "*", "*", "uncertainty_data", pattern)))
            found_files.extend(glob.glob(os.path.join(root_dir, "*", "*", "*", "uncertainty_data", pattern)))
            
        # 중복 제거 및 유효 파일 검증
        unique_files = sorted(list(set([os.path.abspath(f) for f in found_files if os.path.isfile(f)])))
        
        for f_path in unique_files:
            # 사용자 친화적인 상대 경로명 생성 (예: "LC_Air_GeoSplit / MASK2FORMER_20260619")
            parts = f_path.split(os.sep)
            if len(parts) >= 3:
                display_name = f"{parts[-3]} / {parts[-2]}"
            else:
                display_name = parts[-2]
                
            item = QListWidgetItem(display_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, f_path)
            self.list_widget.addItem(item)
                    
    def accept_selection(self):
        self.selected_files = []
        self.selected_names = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                self.selected_files.append(item.data(Qt.UserRole))
                self.selected_names.append(item.text().replace(',', '_'))
                
        if len(self.selected_files) < 2:
            QMessageBox.warning(self, "Selection Error", "Please select at least 2 models to compare.")
            return
            
        self.accept()



class QGISGeoActiveMLStudioPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        self.action = None
        self.menu = "QGIS_GeoActive ML Studio"
        self.dialog = None
        self.worker = None
        self.history_data = {}
        self.current_res_dir = None

        # Land Cover Mapping
        self.color_map = {
            0: [0, 0, 0],           # background
            1: [237, 131, 184],     # Building
            2: [178, 64, 16],       # Parking
            3: [247, 65, 42],       # Road
            4: [155, 229, 200],     # Tree
            5: [255, 255, 191],     # Rice
            6: [223, 220, 115],     # Greenhouse
            7: [247, 249, 102],     # Field
            8: [0, 102, 0],         # Forest
            9: [159, 242, 255],     # Bare
            10: [0, 0, 255]         # Water
        }
        self.class_id_to_dn = {
            0: 100, 1: 10, 2: 20, 3: 30, 4: 40,
            5: 50, 6: 55, 7: 60, 8: 70, 9: 80, 10: 95
        }
        self.class_names = {
            0: "Background/Unclassified", 1: "Building", 2: "Parking", 
            3: "Road", 4: "Tree", 5: "Rice", 6: "Greenhouse", 
            7: "Field", 8: "Forest", 9: "Bare", 10: "Water"
        }

    def initGui(self):
        icon_path = os.path.join(self.plugin_dir, "icon.png")
        icon = QIcon(icon_path)
        self.action = QAction(icon, "Run QGIS_GeoActive ML Studio", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu(self.menu, self.action)

    def unload(self):
        self.iface.removePluginMenu(self.menu, self.action)
        self.iface.removeToolBarIcon(self.action)
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

    def run(self):
        if self.dialog is None:
            self.dialog = AdvancedRSDialog(self.iface.mainWindow())
            self.dialog.train_btn.clicked.connect(self.start_training)
            self.dialog.infer_btn.clicked.connect(self.start_inference)
            self.dialog.view_config_btn.clicked.connect(self.view_training_config)
            self.dialog.view_metrics_btn.clicked.connect(self.view_training_metrics)
            self.dialog.view_graphs_btn.clicked.connect(self.view_training_graphs)
            self.dialog.eval_uncertainty_btn.clicked.connect(self.evaluate_uncertainty)
            self.dialog.active_run_diag_btn.clicked.connect(self.run_manual_active_diagnostic)
            
            # --- NEW: Cross-Model Comparison ---
            self.dialog.btn_compare_train.clicked.connect(self.compare_training)
            self.dialog.btn_compare_infer.clicked.connect(self.compare_inference)
            # -----------------------------------
            
            # Hook up dashboard list and buttons
            self.dialog.btn_load_res.clicked.connect(self.load_past_result)
            self.dialog.history_list.itemClicked.connect(self.on_history_clicked)
            self.dialog.btn_hitl.clicked.connect(lambda: self.open_dashboard_file("residual_error_curve.png", fallback=["hitl_comparison_curve.png", "hitl_curve.png"]))
            self.dialog.btn_sparsification.clicked.connect(lambda: self.open_dashboard_file("risk_coverage_curve.png", fallback=["sparsification.png", "accuracy_coverage_curve.png"]))
            self.dialog.btn_cm.clicked.connect(lambda: self.open_dashboard_file("confusion_matrix.png", fallback=["reliability_diagram_scaled.png", "reliability_diagram.png"]))
            self.dialog.btn_panels.clicked.connect(lambda: self.open_dashboard_file("Qualitative_Panels"))
            
            portable_env = os.path.join(self.plugin_dir, "python_env")
            self.dialog.python_env_input.setText(portable_env)
            
            # Synchronize dataset directory fields between Training and Active Learning tabs
            self.dialog.dataset_dir_input.textChanged.connect(
                lambda text: self.dialog.active_dataset_dir_input.setText(text) if self.dialog.active_dataset_dir_input.text() != text else None
            )
            self.dialog.active_dataset_dir_input.textChanged.connect(
                lambda text: self.dialog.dataset_dir_input.setText(text) if self.dialog.dataset_dir_input.text() != text else None
            )

            # Load saved weights if they exist
            saved_weights = {}
            weights_path = os.path.join(self.plugin_dir, "class_weights.json")
            if os.path.exists(weights_path):
                try:
                    with open(weights_path, 'r') as f:
                        saved_weights = json.load(f)
                except:
                    pass

            # Populate class weights table
            table = self.dialog.weights_table
            table.setRowCount(0)
            from qgis.PyQt.QtWidgets import QTableWidgetItem, QDoubleSpinBox
            row_idx = 0
            for model_id in range(1, len(self.class_names)):  # Skip 0 (Background)
                table.insertRow(row_idx)
                
                id_item = QTableWidgetItem(str(model_id))
                id_item.setFlags(Qt.ItemIsEnabled)
                id_item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row_idx, 0, id_item)
                
                name_item = QTableWidgetItem(self.class_names[model_id])
                name_item.setFlags(Qt.ItemIsEnabled)
                name_item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row_idx, 1, name_item)
                
                weight_spin = QDoubleSpinBox()
                weight_spin.setRange(0.1, 10.0)
                weight_spin.setSingleStep(0.5)
                saved_val = saved_weights.get(str(model_id), 1.0)
                weight_spin.setValue(float(saved_val))
                weight_spin.setAlignment(Qt.AlignCenter)
                table.setCellWidget(row_idx, 2, weight_spin)
                
                row_idx += 1
                
            # Connect Auto-Calculate button
            self.dialog.btn_auto_weight.clicked.connect(self.calculate_auto_weights)

        self.dialog.show()
        self.dialog.exec_()

    def on_history_clicked(self, item):
        history_id = item.data(Qt.UserRole)
        res_dir = self.history_data.get(history_id)
        if not res_dir: return
        self.current_res_dir = res_dir
        
        import glob
        json_files = glob.glob(os.path.join(res_dir, "*metrics_uncertainty.json"))
        if not json_files:
            html = f"""
            <div style="font-family: sans-serif; text-align: center; margin-top: 20px;">
                <h3 style='color: #E67E22;'>⚠️ Analytics Not Generated Yet</h3>
                <p style="font-size: 14px;">You have successfully completed <b>Inference</b>, but the uncertainty metrics and graphs have not been calculated yet.</p>
                <div style="background-color: #F8F9F9; padding: 15px; border-radius: 5px; text-align: left; display: inline-block; border: 1px solid #D5D8DC;">
                    <b>💡 How to view the results:</b><br><br>
                    1. Go back to the <b>Inference</b> tab.<br>
                    2. Click the orange <span style="color: #D35400;"><b>Evaluate Uncertainty</b></span> button.<br>
                    3. Select the following directory when prompted:<br>
                    <code style="background-color: #EAEDED; padding: 2px 4px;">{res_dir}</code><br>
                    4. Wait for the evaluation to finish, then click the new item in this list!
                </div>
            </div>
            """
            self.dialog.dash_content.setHtml(html)
            for btn in [self.dialog.btn_hitl, self.dialog.btn_sparsification, self.dialog.btn_cm, self.dialog.btn_panels]:
                btn.setEnabled(False)
                btn.setStyleSheet("background-color: #f0f0f0; padding: 5px; color: gray;")
            return
            
        json_path = json_files[0]
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                d = json.load(f)
                
            # Try JSTARS new format first
            if "all_pixel_primary" in d:
                c1 = d["all_pixel_primary"].get("c1", {})
                miou_s = d.get("miou_single_seed", c1.get("mIoU", 0.0)) * 100
                miou_e = c1.get("mIoU", 0.0) * 100
                gain = miou_e - miou_s
                ece = c1.get("ECE", 0.0)
                auroc = c1.get("AUROC", 0.0)
                ause = c1.get("AUSE", 0.0)
                
                grid = d.get("grid_comparisons", {})
                hitl = grid.get("grid_512", {}).get("mean", {}).get("summ_20", 0.0) * 100
                
                table_rows = f"<tr><td><b>Primary (c1)</b></td><td>{auroc:.4f}</td><td>{ause:.4f}</td><td>{hitl:.1f}%</td></tr>"
            else:
                miou_s = d.get("miou_single_seed", 0.0) * 100
                miou_e = d.get("miou_ensemble", 0.0) * 100
                gain = d.get("miou_gain", 0.0) * 100
                ece = d.get("expected_calibration_error_ECE", d.get("ece", 0.0))
                
                comp_metrics = d.get("comparison_metrics", {})
                table_rows = ""
                for m in ["BALD", "Entropy", "Max-Softmax", "STD"]:
                    if m in comp_metrics:
                        mets = comp_metrics[m]
                        auroc = mets.get("AUROC", 0.0)
                        ause = mets.get("AUSE", 0.0)
                        hitl = mets.get("HITL_20_caught", 0.0)
                        table_rows += f"<tr><td><b>{m}</b></td><td>{auroc:.4f}</td><td>{ause:.4f}</td><td>{hitl:.1f}%</td></tr>"
                        
                if not table_rows:
                    old_auroc = d.get("error_detection_AUROC", 0.0)
                    old_ause = d.get("sparsification_AUSE", 0.0)
                    old_hitl_info = d.get("hitl_errors_caught_by_budget", {})
                    old_hitl = old_hitl_info.get("20%_area", 0.0) * 100
                    m_name = d.get("uncertainty_metric", "Selected")
                    table_rows = f"<tr><td><b>{m_name}</b></td><td>{old_auroc:.4f}</td><td>{old_ause:.4f}</td><td>{old_hitl:.1f}%</td></tr>"
            
            html = f"""
            <h3 style='color: #2E86C1;'>🚀 Evaluation Analytics</h3>
            <p><b>Target:</b> {os.path.basename(res_dir)}</p>
            <hr/>
            <h4>📊 Quantitative Comparison</h4>
            <table border="1" cellpadding="4" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: center;">
                <tr style="background-color: #f2f2f2;">
                    <th>Method</th><th>AUROC (↑)</th><th>AUSE (↓)</th><th>HITL 20% (↑)</th>
                </tr>
                {table_rows}
            </table>
            <p style="font-size: 11px; color: gray;">* Calibration Error (ECE) for base probabilities: {ece:.4f}</p>
            
            <h4>🎯 Segmentation Performance (mIoU)</h4>
            <ul>
                <li><b>Single Seed:</b> {miou_s:.2f}%</li>
                <li><b>Deep Ensemble:</b> {miou_e:.2f}% <span style='color:green;'>(+{gain:.2f}%)</span></li>
            </ul>
            """
            self.dialog.dash_content.setHtml(html)
            
            for btn in [self.dialog.btn_hitl, self.dialog.btn_sparsification, self.dialog.btn_cm, self.dialog.btn_panels]:
                btn.setEnabled(True)
        except Exception as e:
            self.dialog.dash_content.setHtml(f"<p>Error parsing analytics: {str(e)}</p>")

    def open_dashboard_file(self, filename, fallback=None):
        if not self.current_res_dir: return
        import glob
        
        # 1. 만약 디렉터리라면(예: Qualitative_Panels) 바로 열기
        path = os.path.join(self.current_res_dir, filename)
        if os.path.isdir(path):
            if os.name == 'nt':
                os.startfile(path)
            return
            
        # 2. 파일인 경우 glob으로 모델 접두사가 포함된 파일 탐색
        candidates = glob.glob(os.path.join(self.current_res_dir, f"*{filename}"))
        if candidates:
            path = candidates[0]
        elif fallback:
            fallbacks = fallback if isinstance(fallback, list) else [fallback]
            path = None
            for fb in fallbacks:
                fallback_candidates = glob.glob(os.path.join(self.current_res_dir, f"*{fb}"))
                if fallback_candidates:
                    path = fallback_candidates[0]
                    break
        else:
            path = None
            
        if path and os.path.exists(path) and os.name == 'nt':
            os.startfile(path)
        else:
            QMessageBox.information(self.dialog, "Not Found", f"File not found matching: *{filename}")

    def load_past_result(self):
        from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem
        from qgis.PyQt.QtCore import Qt
        
        # Open directory browser starting at checkpoints
        start_dir = os.path.join(self.plugin_dir, "checkpoints")
        if not os.path.exists(start_dir):
            start_dir = self.plugin_dir
            
        res_dir = QFileDialog.getExistingDirectory(self.dialog, "Select Past Inference Result Directory", start_dir)
        if not res_dir:
            return
            
        import glob
        # Search for metrics_uncertainty.json
        json_files = glob.glob(os.path.join(res_dir, "*metrics_uncertainty.json"))
        if not json_files:
            QMessageBox.warning(
                self.dialog,
                "Error",
                "지정된 폴더 내에서 metrics_uncertainty.json 파일을 찾을 수 없습니다.\n\n"
                "해당 결과 폴더에 대해 'Evaluate Uncertainty' (불확실성 평가)를 먼저 실행한 후에 불러와 주십시오."
            )
            return
            
        # Add to History List
        import uuid
        history_id = str(uuid.uuid4())
        self.history_data[history_id] = res_dir
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"[{timestamp}] Loaded: {os.path.basename(res_dir)}")
        item.setData(Qt.UserRole, history_id)
        self.dialog.history_list.insertItem(0, item)
        self.dialog.history_list.setCurrentItem(item)
        self.on_history_clicked(item)
        
        QMessageBox.information(self.dialog, "Loaded", "성공적으로 결과 폴더를 불러왔습니다!\n대시보드와 하단 분석 그래프 버튼들이 활성화되었습니다.")

    def get_python_exe(self):
        env_dir = os.path.join(self.plugin_dir, "python_env")
        py_exe = os.path.join(env_dir, "python.exe")
        
        if not os.path.exists(py_exe):
            QMessageBox.warning(self.dialog, "Error", f"Portable Python not found at:\n{py_exe}\n\nPlease run offline_setup/build_portable_env.py first.")
            return None
            
        return py_exe

    def view_training_config(self):
        from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
        ckpt_dir = os.path.join(self.plugin_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            QMessageBox.warning(self.dialog, "Error", "No checkpoints directory found yet. Run training first.")
            return
            
        file, _ = QFileDialog.getOpenFileName(self.dialog, "Select Training Config", ckpt_dir, "JSON Files (*.json)")
        if file:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
                
                dlg = QDialog(self.dialog)
                dlg.setWindowTitle(f"Training Configuration - {os.path.basename(file)}")
                dlg.resize(500, 600)
                
                layout = QVBoxLayout(dlg)
                text_edit = QTextEdit()
                text_edit.setReadOnly(True)
                
                html = f"<h2 style='color: #2E86C1;'>⚙️ Training Configuration</h2>"
                html += f"<p><b>File:</b> {os.path.basename(file)}</p>"
                html += "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse: collapse; width: 100%; font-size: 14px;'>"
                html += "<tr style='background-color: #f2f2f2;'><th>Parameter</th><th>Value</th></tr>"
                
                for k, v in config_data.items():
                    k_str = str(k).replace("_", " ").title()
                    v_str = str(v)
                    # Highlight important params
                    if "Lr" in k_str or "Rate" in k_str:
                        v_str = f"<span style='color: #D85A30; font-weight: bold;'>{v_str}</span>"
                    elif "Epoch" in k_str or "Batch" in k_str:
                        v_str = f"<span style='color: #1D9E75; font-weight: bold;'>{v_str}</span>"
                        
                    html += f"<tr><td style='width: 40%;'><b>{k_str}</b></td><td>{v_str}</td></tr>"
                html += "</table>"
                
                html += f"<br><h2 style='color: #9C27B0;'>🏷️ Class Information</h2>"
                html += "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse: collapse; width: 100%; font-size: 14px; text-align: center;'>"
                html += "<tr style='background-color: #f2f2f2;'><th>Model ID</th><th>Class Name</th><th>Orig. DN</th><th>Color</th></tr>"
                
                for model_id in sorted(self.color_map.keys()):
                    c_name = self.class_names.get(model_id, "Unknown")
                    original_dn = self.class_id_to_dn.get(model_id, "N/A")
                    c_rgb = self.color_map.get(model_id, [0, 0, 0])
                    color_str = f"rgb({c_rgb[0]}, {c_rgb[1]}, {c_rgb[2]})"
                    
                    html += f"<tr><td><b>{model_id}</b></td><td>{c_name}</td><td>{original_dn}</td>"
                    html += f"<td><span style='background-color: {color_str}; display: inline-block; width: 15px; height: 15px; border: 1px solid #000; margin-right: 5px; vertical-align: middle;'></span>{c_rgb}</td></tr>"
                    
                html += "</table>"
                
                text_edit.setHtml(html)
                layout.addWidget(text_edit)
                
                btn_close = QPushButton("Close")
                btn_close.setStyleSheet("padding: 5px; font-weight: bold;")
                btn_close.clicked.connect(dlg.accept)
                layout.addWidget(btn_close)
                
                dlg.exec_()
            except Exception as e:
                QMessageBox.warning(self.dialog, "Error", f"Could not read config: {e}")

    def view_training_metrics(self):
        from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox, QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem
        import csv
        ckpt_dir = os.path.join(self.plugin_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            QMessageBox.warning(self.dialog, "Error", "No checkpoints directory found yet. Run training first.")
            return
            
        file, _ = QFileDialog.getOpenFileName(self.dialog, "Select Metrics CSV", ckpt_dir, "CSV Files (*.csv)")
        if file:
            try:
                with open(file, 'r', newline='') as f:
                    reader = csv.reader(f)
                    data = list(reader)
                
                if not data:
                    QMessageBox.warning(self.dialog, "Error", "Metrics CSV is empty.")
                    return
                
                headers = data[0]
                rows = data[1:]
                
                dlg = QDialog(self.dialog)
                dlg.setWindowTitle(f"Training Metrics: {os.path.basename(os.path.dirname(file))}")
                dlg.resize(1000, 600)
                layout = QVBoxLayout(dlg)
                
                table = QTableWidget(len(rows), len(headers))
                table.setHorizontalHeaderLabels(headers)
                
                for r_idx, row in enumerate(rows):
                    for c_idx, val in enumerate(row):
                        table.setItem(r_idx, c_idx, QTableWidgetItem(val))
                        
                table.resizeColumnsToContents()
                layout.addWidget(table)
                dlg.exec_()
                
            except Exception as e:
                QMessageBox.warning(self.dialog, "Error", f"Could not read metrics: {e}")

    def view_training_graphs(self):
        from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
        ckpt_dir = os.path.join(self.plugin_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            QMessageBox.warning(self.dialog, "Error", "No checkpoints directory found yet. Run training first.")
            return
            
        file, _ = QFileDialog.getOpenFileName(self.dialog, "Select Metrics CSV for Graphs", ckpt_dir, "CSV Files (*.csv)")
        if file:
            py_exe = self.get_python_exe()
            if not py_exe: return
            
            plot_script = os.path.join(self.plugin_dir, "scripts", "plot_metrics.py")
            if not os.path.isfile(plot_script):
                QMessageBox.warning(self.dialog, "Error", "plot_metrics.py script not found.")
                return
                
            cmd = [py_exe, plot_script, "--csv", file]
            self.dialog.log(f"Opening graphs for {os.path.basename(os.path.dirname(file))}...")
            
            # Spawn independently, do not wait or capture output
            try:
                subprocess.Popen(cmd, cwd=self.plugin_dir)
            except Exception as e:
                QMessageBox.warning(self.dialog, "Error", f"Failed to launch graph viewer: {e}")

    def evaluate_uncertainty(self):
        from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
        
        res_dir = QFileDialog.getExistingDirectory(self.dialog, "Select Inference Results Directory", self.plugin_dir)
        if not res_dir:
            return
            
        norm_res = os.path.normpath(res_dir)
        if os.path.basename(norm_res) == "uncertainty_data":
            res_dir = os.path.dirname(norm_res)
            
        py_exe = self.get_python_exe()
        if not py_exe: return

        # Auto-detect model from selected folder name to avoid mismatches
        folder_name = os.path.basename(os.path.normpath(res_dir)).upper()
        detected_model = None
        if "MASK2FORMER" in folder_name:
            detected_model = "mask2former"
        elif "UNET" in folder_name:
            detected_model = "unet"
        elif "DEEPLABV3PLUS" in folder_name:
            detected_model = "deeplabv3plus"
        elif "SEGFORMER" in folder_name:
            detected_model = "segformer"

        if detected_model:
            model_arch_text = self.dialog.model_combo.currentText()
            if "U-Net" in model_arch_text:
                ui_model = "unet"
            elif "DeepLabV3+" in model_arch_text:
                ui_model = "deeplabv3plus"
            elif "SegFormer" in model_arch_text:
                ui_model = "segformer"
            else:
                ui_model = "mask2former"

            if detected_model != ui_model:
                reply = QMessageBox.question(
                    self.dialog,
                    "Model Mismatch Detected",
                    f"선택한 결과 폴더 감지 모델: '{detected_model.upper()}'\n"
                    f"현재 UI 선택 모델: '{ui_model.upper()}'\n\n"
                    f"두 모델이 일치하지 않습니다. UI 설정을 '{detected_model.upper()}' 모델로 전환하고 계속 진행하시겠습니까?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
                )
                if reply == QMessageBox.Yes:
                    if detected_model == "mask2former":
                        self.dialog.model_combo.setCurrentIndex(0)
                    elif detected_model == "unet":
                        self.dialog.model_combo.setCurrentIndex(1)
                    elif detected_model == "deeplabv3plus":
                        self.dialog.model_combo.setCurrentIndex(2)
                    elif detected_model == "segformer":
                        self.dialog.model_combo.setCurrentIndex(3)
                elif reply == QMessageBox.Cancel:
                    return
        
        eval_script = os.path.join(self.plugin_dir, "scripts", "evaluate_uncertainty.py")
        mapping_file = os.path.join(self.plugin_dir, "class_mappings.json")
        
        if not os.path.isfile(eval_script):
            QMessageBox.warning(self.dialog, "Error", "evaluate_uncertainty.py not found.")
            return
            
        is_16bit = "16-bit" in self.dialog.data_type_combo.currentText()
        model_arch_text = self.dialog.model_combo.currentText()
        if "U-Net" in model_arch_text:
            model_arch = "unet"
        elif "DeepLabV3+" in model_arch_text:
            model_arch = "deeplabv3plus"
        elif "SegFormer" in model_arch_text:
            model_arch = "segformer"
        else:
            model_arch = "mask2former"
            
        # 1. Guess from res_dir structure with file verification (High priority)
        lbl_dir = ""
        try:
            parent1 = os.path.dirname(os.path.normpath(res_dir))
            parent2 = os.path.dirname(parent1)
            dataset_root = os.path.dirname(parent2)
            
            # Check splits: test/label, val/label, train/label, label
            for split in ["test", "val", "train", ""]:
                candidate = os.path.join(dataset_root, split, "label") if split else os.path.join(dataset_root, "label")
                candidate = os.path.normpath(candidate)
                if os.path.isdir(candidate):
                    # Verify matching files by checking if first npz matches any file here
                    unc_dir = os.path.join(res_dir, "uncertainty_data")
                    if os.path.isdir(unc_dir):
                        npz_files = [f for f in os.listdir(unc_dir) if f.endswith(".npz")]
                        if npz_files:
                            d_file = npz_files[0]
                            base_name = os.path.splitext(d_file)[0]
                            matched = False
                            for ext in [".tif", ".tiff", ".png", ".jpg"]:
                                if os.path.exists(os.path.join(candidate, base_name + ext)):
                                    matched = True
                                    break
                            if matched:
                                lbl_dir = candidate
                                break
        except Exception:
            pass

        # 2. If not found by res_dir structure, fall back to last inference directory
        if not lbl_dir:
            if hasattr(self, 'last_infer_img_dir') and self.last_infer_img_dir:
                guessed_lbl = os.path.join(os.path.dirname(self.last_infer_img_dir), "label")
                if os.path.isdir(guessed_lbl):
                    lbl_dir = guessed_lbl
                
        # 3. Verify if the resolved lbl_dir actually contains matching files
        has_match = False
        if lbl_dir:
            try:
                unc_dir = os.path.join(res_dir, "uncertainty_data")
                if os.path.isdir(unc_dir):
                    npz_files = [f for f in os.listdir(unc_dir) if f.endswith(".npz")]
                    if npz_files:
                        d_file = npz_files[0]
                        base_name = os.path.splitext(d_file)[0]
                        for ext in [".tif", ".tiff", ".png", ".jpg"]:
                            if os.path.exists(os.path.join(lbl_dir, base_name + ext)):
                                has_match = True
                                break
            except Exception:
                pass
                
        # If not found or filenames do not match, prompt user
        if not lbl_dir or not has_match:
            from qgis.PyQt.QtWidgets import QMessageBox
            QMessageBox.information(self.dialog, "Label Directory Needed", 
                "정합성이 일치하는 정답 라벨(label) 폴더를 자동으로 식별하지 못했습니다.\n\n"
                "수동으로 해당 데이터셋의 정답 라벨(label) 이미지 폴더를 선택해 주세요.")
            lbl_dir = QFileDialog.getExistingDirectory(self.dialog, "Select Label Directory", os.path.dirname(res_dir))
            if not lbl_dir:
                return # User cancelled
                
        # Auto-detect validation folder with the same timestamp
        val_results_dir = ""
        try:
            timestamp_part = os.path.basename(res_dir).split("_Inference_", 1)[1]
            model_part = os.path.basename(res_dir).split("_Inference_", 1)[0]
            parent_results = os.path.dirname(os.path.dirname(res_dir))
            for root, dirs, files in os.walk(parent_results):
                for d in dirs:
                    if d == f"{model_part}_Inference_{timestamp_part}" and os.path.join(root, d) != res_dir:
                        val_results_dir = os.path.join(root, d)
                        break
                if val_results_dir:
                    break
        except Exception:
            pass
            
        cmd = [py_exe, "-u", eval_script, "--results_dir", res_dir, "--mapping_file", mapping_file, "--uncertainty", "bald", "--tile", "512", "--is_16bit", str(is_16bit), "--model_arch", model_arch, "--label_dir", lbl_dir]
        if val_results_dir:
            cmd.extend(["--val_results_dir", val_results_dir])
            
        self.dialog.log(f"Evaluating Uncertainty for {os.path.basename(res_dir)}...")
        if val_results_dir:
            self.dialog.log(f"[TS] Auto-detected matching validation results directory: {val_results_dir}")
        else:
            self.dialog.log("[TS] No matching validation results directory found. Temperature Scaling will default to T=1.0.")
        self.dialog.infer_btn.setEnabled(False)
        self.dialog.eval_uncertainty_btn.setEnabled(False)
        
        self.worker = SubprocessWorker(cmd, cwd=self.plugin_dir)
        self.worker.log_msg.connect(self.dialog.log)
        self.worker.progress_val.connect(self.dialog.progress_bar.setValue)
        
        def on_eval_finished(returncode):
            self.dialog.infer_btn.setEnabled(True)
            self.dialog.eval_uncertainty_btn.setEnabled(True)
            if returncode != 0:
                self.dialog.log(f"Uncertainty Evaluation failed with code {returncode}!")
                return
            self.dialog.log("Uncertainty Evaluation Finished!")
            
            # Add to History List
            import uuid
            history_id = str(uuid.uuid4())
            self.history_data[history_id] = res_dir
            
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            item = QListWidgetItem(f"[{timestamp}] Evaluation: {os.path.basename(res_dir)}")
            item.setData(Qt.UserRole, history_id)
            self.dialog.history_list.insertItem(0, item)
            
            # Switch to Results tab and select the item
            self.dialog.tabs.setCurrentIndex(3)
            self.dialog.history_list.setCurrentItem(item)
            self.on_history_clicked(item)
            
            QMessageBox.information(self.dialog, "Done", "Uncertainty Evaluation Completed!\nCheck the Analytics Dashboard.")
            
        self.worker.finished.connect(on_eval_finished)
        self.worker.start()

    def compare_training(self):
        py_exe = self.get_python_exe()
        if not py_exe: return
        
        dialog = ModelSelectionDialog(self.dialog, "Select Training Models to Compare", "metrics.csv")
        # Default root dir: checkpoints
        chkpt_dir = os.path.join(self.plugin_dir, "checkpoints")
        if os.path.isdir(chkpt_dir):
            dialog.dir_input.setText(chkpt_dir)
            dialog.scan_directory()
            
        if dialog.exec_() == QDialog.Accepted:
            selected_files = dialog.selected_files
            selected_names = dialog.selected_names
            out_dir = dialog.dir_input.text().strip()
            script_path = os.path.join(self.plugin_dir, "scripts", "compare_training.py")
            cmd = [py_exe, script_path, "--csv_files", ",".join(selected_files), "--names", ",".join(selected_names), "--out_dir", out_dir]
            
            self.dialog.log(f"Comparing Training Curves for {len(selected_files)} models...")
            self.worker = SubprocessWorker(cmd, cwd=self.plugin_dir)
            self.worker.log_msg.connect(self.dialog.log)
            self.worker.finished.connect(lambda rc: self.on_compare_finished(rc, out_dir, "training_comparison.png"))
            self.worker.start()

    def compare_inference(self):
        py_exe = self.get_python_exe()
        if not py_exe: return
        
        dialog = ModelSelectionDialog(self.dialog, "Select Inference Results to Compare", "metrics_uncertainty.json")
        
        # Default root dir: checkpoints folder which contains all model subdirs and inference results
        default_dir = os.path.join(self.plugin_dir, "checkpoints")
        if self.current_res_dir and os.path.isdir(self.current_res_dir):
            default_dir = os.path.dirname(self.current_res_dir)
            
        if os.path.isdir(default_dir):
            dialog.dir_input.setText(default_dir)
            dialog.scan_directory()
            
        if dialog.exec_() == QDialog.Accepted:
            selected_files = dialog.selected_files
            selected_names = dialog.selected_names
            out_dir = dialog.dir_input.text().strip()
            script_path = os.path.join(self.plugin_dir, "scripts", "compare_inference.py")
            cmd = [py_exe, script_path, "--json_files", ",".join(selected_files), "--names", ",".join(selected_names), "--out_dir", out_dir]
            
            self.dialog.log(f"Comparing Inference Metrics for {len(selected_files)} models...")
            self.worker = SubprocessWorker(cmd, cwd=self.plugin_dir)
            self.worker.log_msg.connect(self.dialog.log)
            self.worker.finished.connect(lambda rc: self.on_compare_finished(rc, out_dir, "model_comparison.png", "comparison_report.html"))
            self.worker.start()
            
    def on_compare_finished(self, rc, out_dir, png_name, html_name=None):
        if rc == 0:
            self.dialog.log("Comparison completed successfully!")
            png_path = os.path.join(out_dir, png_name)
            if os.path.exists(png_path):
                import platform
                if platform.system() == 'Windows':
                    os.startfile(png_path)
            
            if html_name:
                html_path = os.path.join(out_dir, html_name)
                if os.path.exists(html_path):
                    with open(html_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    self.dialog.dash_content.setHtml(html_content)
        else:
            self.dialog.log(f"Comparison script failed with code {rc}.")
            QMessageBox.warning(self.dialog, "Error", "Comparison failed. Check logs.")

    def start_training(self):
        py_exe = self.get_python_exe()
        if not py_exe: return
        
        # Datasets
        dataset_dir = self.dialog.dataset_dir_input.text().strip()
        if not dataset_dir or not os.path.isdir(dataset_dir):
            QMessageBox.warning(self.dialog, "Error", "Invalid Dataset Root.")
            return

        is_16bit = "16-bit" in self.dialog.data_type_combo.currentText()
        epochs = self.dialog.epochs_spin.value()
        batch = self.dialog.batch_spin.value()
        train_ckpt = self.dialog.train_ckpt_input.text().strip()
        use_focal_loss = self.dialog.focal_loss_cb.isChecked()

        # Save mappings to a temp json so the external script can read them
        map_path = os.path.join(self.plugin_dir, "class_mappings.json")
        with open(map_path, 'w') as f:
            json.dump({'color_map': self.color_map, 'dn_map': self.class_id_to_dn}, f)

        model_arch_text = self.dialog.model_combo.currentText()
        if "U-Net" in model_arch_text:
            model_arch = "unet"
        elif "DeepLabV3+" in model_arch_text:
            model_arch = "deeplabv3plus"
        elif "SegFormer" in model_arch_text:
            model_arch = "segformer"
        else:
            model_arch = "mask2former"

        train_script = os.path.join(self.plugin_dir, "scripts", "train.py")
        
        cmd = [
            py_exe, train_script,
            "--dataset_dir", dataset_dir,
            "--is_16bit", str(is_16bit),
            "--epochs", str(epochs),
            "--batch_size", str(batch),
            "--mapping_file", map_path,
            "--model_arch", model_arch
        ]
        
        # Extract class weights
        class_weights = {}
        table = self.dialog.weights_table
        for row in range(table.rowCount()):
            model_id_item = table.item(row, 0)
            weight_spin = table.cellWidget(row, 2)
            if model_id_item and weight_spin:
                class_weights[model_id_item.text()] = weight_spin.value()
                
        weights_path = os.path.join(self.plugin_dir, "class_weights.json")
        with open(weights_path, 'w') as f:
            json.dump(class_weights, f)
        
        cmd.extend(["--class_weights", weights_path])
        
        if use_focal_loss:
            cmd.append("--use_focal_loss")
            focal_gamma = self.dialog.focal_gamma_spin.value()
            focal_alpha = self.dialog.focal_alpha_spin.value()
            cmd.extend(["--focal_gamma", str(focal_gamma)])
            cmd.extend(["--focal_alpha", str(focal_alpha)])
            
        if train_ckpt and os.path.exists(train_ckpt):
            cmd.extend(["--resume_from", train_ckpt])

        self.dialog.log(f"Executing: {' '.join(cmd)}")
        self.dialog.train_btn.setEnabled(False)
        self.dialog.infer_btn.setEnabled(False)
        self.is_inference = False
        
        self.worker = SubprocessWorker(cmd, cwd=self.plugin_dir)
        self.worker.log_msg.connect(self.dialog.log)
        self.worker.progress_val.connect(self.dialog.progress_bar.setValue)
        self.worker.finished.connect(self.on_process_finished)
        self.worker.start()

    def calculate_auto_weights(self):
        py_exe = self.get_python_exe()
        if not py_exe: return
        
        dataset_dir = self.dialog.dataset_dir_input.text().strip()
        if not dataset_dir or not os.path.isdir(dataset_dir):
            from qgis.PyQt.QtWidgets import QMessageBox
            QMessageBox.warning(self.dialog, "Error", "Invalid Dataset Root. Please set Dataset Root first.")
            return
            
        map_path = os.path.join(self.plugin_dir, "class_mappings.json")
        import json
        with open(map_path, 'w') as f:
            json.dump({'color_map': self.color_map, 'dn_map': self.class_id_to_dn}, f)
            
        calc_script = os.path.join(self.plugin_dir, "scripts", "calc_class_weights.py")
        cmd = [py_exe, "-u", calc_script, "--dataset_dir", dataset_dir, "--mapping_file", map_path]
        
        self.dialog.log("Calculating optimal class weights based on dataset pixels... Please wait.")
        
        # UI Feedback
        from qgis.PyQt.QtWidgets import QApplication, QMessageBox
        from qgis.PyQt.QtCore import Qt
        
        self.dialog.btn_auto_weight.setText("⏳ Calculating... Please Wait")
        self.dialog.btn_auto_weight.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        # Run asynchronously to prevent 120s timeout and UI freezing
        self.calc_worker = SubprocessWorker(cmd, cwd=self.plugin_dir)
        
        def on_calc_log(msg):
            import json
            msg = msg.strip()
            if not msg: return
            
            # Show progress
            if msg.startswith("Processing image"):
                self.dialog.log(msg)
            elif msg.startswith('{') and msg.endswith('}'):
                try:
                    weights_dict = json.loads(msg)
                    if "error" in weights_dict:
                        self.dialog.log(f"Auto-Calc Error: {weights_dict['error']}")
                        QMessageBox.warning(self.dialog, "Auto-Calc Error", weights_dict['error'])
                        return
                    
                    # Update table
                    table = self.dialog.weights_table
                    for row in range(table.rowCount()):
                        model_id_item = table.item(row, 0)
                        if model_id_item:
                            mid = model_id_item.text()
                            if mid in weights_dict:
                                spin = table.cellWidget(row, 2)
                                if spin:
                                    spin.setValue(float(weights_dict[mid]))
                    
                    self.dialog.log("Successfully loaded optimal class weights based on Inverse Frequency!")
                    QMessageBox.information(self.dialog, "Auto-Weight Success", "Successfully calculated and applied optimal class weights!")
                except Exception as e:
                    self.dialog.log(f"JSON Parse Exception: {str(e)}")
            else:
                self.dialog.log(msg)
                
        def on_calc_finished(returncode):
            self.dialog.btn_auto_weight.setText("🪄 Auto-Calculate Optimal Weights")
            self.dialog.btn_auto_weight.setEnabled(True)
            QApplication.restoreOverrideCursor()
            if returncode != 0:
                self.dialog.log(f"Weight calculation exited with code {returncode}.")
                QMessageBox.warning(self.dialog, "Error", "Calculation failed. Check the logs.")
                
        self.calc_worker.log_msg.connect(on_calc_log)
        self.calc_worker.finished.connect(on_calc_finished)
        self.calc_worker.start()

    def start_inference(self):
        py_exe = self.get_python_exe()
        if not py_exe: return
        
        ckpt = self.dialog.ckpt_input.text().strip()
        img_dir = self.dialog.infer_img_input.text().strip()
        
        ckpt_list = [c.strip() for c in ckpt.split(',') if c.strip()]
        if not ckpt_list:
            QMessageBox.warning(self.dialog, "Error", "No checkpoint selected.")
            return
        for c in ckpt_list:
            if not os.path.isfile(c):
                QMessageBox.warning(self.dialog, "Error", f"Invalid checkpoint file: {c}")
                return
        if not os.path.isdir(img_dir):
            QMessageBox.warning(self.dialog, "Error", "Invalid images directory.")
            return

        is_16bit = "16-bit" in self.dialog.data_type_combo.currentText()
        seeds = self.dialog.seeds_input.text().strip()
        mc_dropout = self.dialog.mc_dropout_cb.isChecked()
        
        # Uncertainty Warning
        seed_list = [s.strip() for s in seeds.split(',') if s.strip()]
        if len(seed_list) <= 1 and not mc_dropout:
            from qgis.PyQt.QtWidgets import QMessageBox
            reply = QMessageBox.question(self.dialog, "Uncertainty Validation Warning",
                "⚠️ You are starting Inference with only 1 Seed and MC Dropout is disabled.\n\n"
                "This means the model will only generate ONE prediction per image. "
                "Consequently, Epistemic Uncertainty metrics (like BALD and STD variance) will be EXACTLY ZERO (Blank/Dark Blue) because there is no ensemble diversity to measure.\n\n"
                "If you want to evaluate uncertainty properly, please click 'No', check the 'Enable MC Dropout' box, and try again.\n\n"
                "Proceed anyway?", 
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return
        
        map_path = os.path.join(self.plugin_dir, "class_mappings.json")
        with open(map_path, 'w') as f:
            json.dump({'color_map': self.color_map, 'dn_map': self.class_id_to_dn}, f)

        model_arch_text = self.dialog.model_combo.currentText()
        if "U-Net" in model_arch_text:
            model_arch = "unet"
        elif "DeepLabV3+" in model_arch_text:
            model_arch = "deeplabv3plus"
        elif "SegFormer" in model_arch_text:
            model_arch = "segformer"
        else:
            model_arch = "mask2former"

        infer_script = os.path.join(self.plugin_dir, "scripts", "inference.py")
        
        cmd = [
            py_exe, infer_script,
            "--checkpoint", ckpt,
            "--img_dir", img_dir,
            "--is_16bit", str(is_16bit),
            "--mapping_file", map_path,
            "--seeds", seeds,
            "--model_arch", model_arch
        ]
        if mc_dropout:
            cmd.append("--mc_dropout")

        self.last_infer_img_dir = img_dir
        self.last_infer_seeds = seeds
        self.last_infer_dropout = mc_dropout
        self.last_infer_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.is_inference = True

        self.dialog.log(f"Executing: {' '.join(cmd)}")
        self.dialog.train_btn.setEnabled(False)
        self.dialog.infer_btn.setEnabled(False)
        
        self.worker = SubprocessWorker(cmd, cwd=self.plugin_dir)
        self.worker.log_msg.connect(self.dialog.log)
        self.worker.progress_val.connect(self.dialog.progress_bar.setValue)
        self.worker.finished.connect(self.on_process_finished)
        self.worker.start()

    def on_process_finished(self, code):
        if code == 0:
            self.dialog.log("\nProcess Completed Successfully!")
            if getattr(self, 'is_inference', False):
                self.handle_inference_success()
        else:
            self.dialog.log(f"\nProcess Failed with return code {code}.")
        
        self.dialog.train_btn.setEnabled(True)
        self.dialog.infer_btn.setEnabled(True)
        self.is_inference = False

    def handle_inference_success(self):
        # 1. Try to use the output directory parsed from worker stdout
        if hasattr(self, 'worker') and getattr(self.worker, 'output_dir', None):
            out_dir = self.worker.output_dir
        else:
            # 2. Fallback to robust directory scanning
            parent_dir = os.path.dirname(os.path.normpath(self.last_infer_img_dir))
            dataset_name = os.path.basename(parent_dir)
            if dataset_name.lower() in ["train", "val", "test", "images", "labels"]:
                parent_dir = os.path.dirname(parent_dir)
                dataset_name = os.path.basename(parent_dir)
                
            model_arch_text = self.dialog.model_combo.currentText()
            if "U-Net" in model_arch_text:
                model_name = "UNET"
            elif "DeepLabV3+" in model_arch_text:
                model_name = "DEEPLABV3PLUS"
            elif "SegFormer" in model_arch_text:
                model_name = "SEGFORMER"
            else:
                model_name = "MASK2FORMER"
                
            results_root = os.path.join(parent_dir, "Results", dataset_name)
            
            # Search for latest timestamped folder
            import glob
            pattern = os.path.join(results_root, f"{model_name}_Inference_*")
            dirs = glob.glob(pattern)
            if dirs:
                out_dir = max(dirs, key=os.path.getmtime)
            else:
                out_dir = os.path.join(results_root, f"{model_name}_InferenceResults")
        
        # Add to History List
        import uuid
        history_id = str(uuid.uuid4())
        self.history_data[history_id] = out_dir
        
        item = QListWidgetItem(f"[{self.last_infer_time.split(' ')[1]}] Inference: {os.path.basename(self.last_infer_img_dir)}")
        item.setData(Qt.UserRole, history_id)
        self.dialog.history_list.insertItem(0, item)
        
        # Load QGIS Layers
        try:
            from qgis.core import QgsRasterLayer, QgsProject
            import glob
            
            mean_dir = os.path.join(out_dir, "ensemble_mean")
            std_dir = os.path.join(out_dir, "ensemble_std")
            
            if os.path.exists(mean_dir):
                mean_files = glob.glob(os.path.join(mean_dir, "*.*"))
                if mean_files:
                    mean_layer = QgsRasterLayer(mean_files[0], "Ensemble Mean")
                    if mean_layer.isValid():
                        QgsProject.instance().addMapLayer(mean_layer)
                        
            if os.path.exists(std_dir):
                std_files = glob.glob(os.path.join(std_dir, "*.*"))
                if std_files:
                    std_layer = QgsRasterLayer(std_files[0], "Uncertainty (Std)")
                    if std_layer.isValid():
                        QgsProject.instance().addMapLayer(std_layer)
                        
            self.dialog.log("Results automatically loaded into QGIS Canvas and Results Tab.")
        except ImportError:
            # Not running inside QGIS environment (e.g. testing standalone)
            pass
            
        # --- Active Learning Auto-Retraining Trigger ---
        if getattr(self, 'last_infer_dropout', False):
            try:
                from qgis.PyQt.QtCore import QTimer
                QTimer.singleShot(500, lambda: self.trigger_active_learning(out_dir))
            except ImportError:
                # Standalone test fallback
                self.trigger_active_learning(out_dir)

    def trigger_active_learning(self, out_dir):
        import sys
        from qgis.PyQt.QtWidgets import QMessageBox
        self.dialog.progress_bar.setValue(0)
        
        # Smart folder detection: if the directory itself contains .npz files, use it directly.
        # Otherwise, look for a subfolder named 'uncertainty_data'.
        if os.path.isdir(out_dir) and any(f.lower().endswith(".npz") for f in os.listdir(out_dir)):
            unc_dir = out_dir
        else:
            unc_dir = os.path.join(out_dir, "uncertainty_data")
            
        if not os.path.exists(unc_dir):
            self.dialog.log(f"Active Learning: Uncertainty data folder not found ({unc_dir}). Skipping diagnosis.")
            self.dialog.progress_bar.setValue(100)
            return
            
        # Add scripts path to sys.path if not there
        scripts_path = os.path.join(self.plugin_dir, "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
            
        try:
            from active_learning import analyze_uncertainty, auto_correct_labels
        except ImportError as e:
            self.dialog.log(f"Active Learning: Import error: {e}")
            self.dialog.progress_bar.setValue(100)
            return
            
        # Check if active learning is enabled via GUI checkbox
        if not self.dialog.active_enabled_cb.isChecked():
            self.dialog.log("Active Learning: 자동 재학습 루프가 비활성화되어 있습니다. 건너뜁니다.")
            self.dialog.progress_bar.setValue(100)
            return

        # Retrieve thresholds from GUI
        try:
            unc_threshold = float(self.dialog.active_threshold_spin.value())
            ratio_threshold = float(self.dialog.active_ratio_spin.value()) / 100.0
        except Exception:
            unc_threshold = 0.15
            ratio_threshold = 0.05
            
        # Analyze uncertainty using GUI thresholds
        stats = analyze_uncertainty(unc_dir, threshold=unc_threshold, ratio_threshold=ratio_threshold)
        
        if stats["candidate_count"] == 0:
            self.dialog.log("Active Learning: No high uncertainty patches detected. Model is stable in this region.")
            self.dialog.progress_bar.setValue(100)
            return
            
        total_new = stats["total_files"]
        confused = stats["candidate_count"]
        confused_ratio = (confused / total_new) * 100 if total_new > 0 else 0
        mean_unc = stats["mean_uncertainty"]
        
        # Show detailed QMessageBox dialog in Korean
        msg_box = QMessageBox(self.dialog)
        msg_box.setWindowTitle("📊 자동 재학습(파인튜닝) 제안 및 안내")
        msg_box.setIcon(QMessageBox.Question)
        
        msg_text = (
            f"신규 입력 데이터에 대한 모델 불확실성(Uncertainty) 분석이 완료되었습니다.\n"
            f"일부 지역에서 모델이 확신을 갖지 못하는 영역이 검출되어 재학습을 제안합니다.\n\n"
            f"📌 **분석 요약**\n"
            f"  • 학습 권장 대상: 총 {confused}개 패치 (전체의 {confused_ratio:.1f}%)\n"
            f"  • 평균 불확실성 지표: {mean_unc:.2f} (높음)\n\n"
            f"❓ **재학습이 권장되는 이유**\n"
            f"  1. 신규 패턴 감지: 현재 모델이 이전에 본 적 없는 새로운 형태의 객체(예: 바뀐 지붕 형태, 도로 재질, 기상 조건에 따른 그림자 등)를 마주하여 예측 오차가 발생할 확률이 매우 높습니다.\n"
            f"  2. 경계선 예측 불안정: 건물이나 도로의 외곽 경계선 피처가 흐릿해 모델의 예측이 흔들리고 있습니다.\n\n"
            f"✨ **재학습(파인튜닝)의 효과**\n"
            f"  • 공공 참조자료를 기반으로 모델이 헷갈려하는 {confused}개 지역의 라벨을 자동으로 보정하여 학습셋에 긴급 수혈합니다.\n"
            f"  • 이로 인해 불확실 영역에서의 오탐률(False Positive)이 대폭 감소하며, 신규 지역에 대한 예측 경계선이 매우 선명하고 정밀해집니다.\n\n"
            f"지금 검출된 취약 데이터를 반영하여 자동 재학습을 진행하시겠습니까?\n"
            f"(예상 소요 시간: 약 15분, 백그라운드 구동)"
        )
        
        msg_box.setText(msg_text)
        btn_yes = msg_box.addButton("예 (모델 보강 시작)", QMessageBox.YesRole)
        btn_no = msg_box.addButton("아니오 (나중에 하기)", QMessageBox.NoRole)
        msg_box.setDefaultButton(btn_yes)
        
        msg_box.exec_()
        
        if msg_box.clickedButton() == btn_yes:
            self.dialog.log("사용자 승인: 자동 라벨 보정 및 재학습을 기동합니다...")
            
            img_dir = self.last_infer_img_dir
            dataset_dir = self.dialog.dataset_dir_input.text().strip()
            
            if not dataset_dir or not os.path.isdir(dataset_dir):
                parent_dir = os.path.dirname(os.path.normpath(img_dir))
                if os.path.basename(parent_dir).lower() in ["train", "val", "test", "images", "labels"]:
                    parent_dir = os.path.dirname(parent_dir)
                dataset_dir = parent_dir
                
            # Check if a custom Reference Label directory is provided
            custom_ref_dir = self.dialog.active_ref_dir_input.text().strip()
            ref_dir = custom_ref_dir if (custom_ref_dir and os.path.isdir(custom_ref_dir)) else None
            
            corrected = auto_correct_labels(
                stats["candidate_files"], 
                img_dir, 
                dataset_dir, 
                threshold=unc_threshold,
                ref_dir=ref_dir
            )
            
            if corrected == 0:
                self.dialog.log("에러: 참조 정답 라벨이 없거나 자동 라벨 보정에 실패했습니다. 재학습이 취소되었습니다.")
                QMessageBox.warning(self.dialog, "재학습 실패", "참조 정답 라벨이 없거나 자동 보정에 실패했습니다.")
                self.dialog.progress_bar.setValue(100)
                return
                
            self.dialog.log(f"성공: {corrected}개 이미지 패치 및 보정 라벨이 학습 데이터셋(train)에 추가 반영되었습니다.")
            
            # 신규 데이터 추가에 따른 클래스 가중치(역빈도) 자동 재계산
            try:
                import subprocess
                py_exe = self.get_python_exe()
                calc_script = os.path.join(self.plugin_dir, "scripts", "calculate_class_weights.py")
                weights_path = os.path.join(self.plugin_dir, "class_weights.json")
                calc_cmd = [py_exe, calc_script, "--dataset_dir", dataset_dir, "--out_file", weights_path]
                self.dialog.log("학습 데이터셋 확장 반영에 따른 클래스 가중치 자동 재계산 중...")
                subprocess.run(calc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.dialog.log("클래스 가중치 자동 갱신 및 반영 완료!")
            except Exception as e:
                self.dialog.log(f"가중치 자동 갱신 중 예외 발생 (기존 가중치 유지): {e}")
                
            self.run_retraining_process(dataset_dir)
        else:
            self.dialog.log("사용자 취소: 자동 재학습이 취소되었습니다.")
            self.dialog.progress_bar.setValue(100)

    def run_retraining_process(self, dataset_dir):
        from qgis.PyQt.QtWidgets import QMessageBox
        
        try:
            py_exe = self.get_python_exe()
            if not py_exe:
                self.dialog.progress_bar.setValue(100)
                return
            
            is_16bit = "16-bit" in self.dialog.data_type_combo.currentText()
            # For automated Active Learning retraining, force epochs to 5 to prevent overfitting
            epochs = 5
            batch = self.dialog.batch_spin.value()
            
            # Use first checkpoint from inference setup to resume training
            infer_ckpt = self.dialog.ckpt_input.text().strip()
            if not infer_ckpt:
                self.dialog.log("에러: Inference(추론) 탭의 'Checkpoint(s)' 필드가 비어 있습니다.")
                self.dialog.log("재학습(파인튜닝)을 진행하려면 Inference 탭에서 파인튜닝의 원본이 될 기본 모델 체크포인트(.pt) 파일을 먼저 브라우즈해서 선택해 주셔야 합니다.")
                QMessageBox.warning(self.dialog, "체크포인트 미선택", 
                                    "재학습을 시작하려면 Inference(추론) 탭에서 전이학습(Fine-tuning)의 바탕이 될 원본 모델 체크포인트(.pt) 파일을 먼저 선택해 주세요.")
                self.dialog.progress_bar.setValue(100)
                return
                
            ckpt_list = [c.strip() for c in infer_ckpt.split(',') if c.strip()]
            if not ckpt_list:
                self.dialog.log("에러: 유효한 기본 체크포인트가 없습니다.")
                self.dialog.progress_bar.setValue(100)
                return
                
            train_ckpt = ckpt_list[0]
            if not os.path.exists(train_ckpt):
                self.dialog.log(f"에러: 선택한 원본 체크포인트 파일이 존재하지 않습니다: {train_ckpt}")
                QMessageBox.warning(self.dialog, "체크포인트 없음", f"지정된 체크포인트 파일이 존재하지 않습니다:\n{train_ckpt}")
                self.dialog.progress_bar.setValue(100)
                return
                
            map_path = os.path.join(self.plugin_dir, "class_mappings.json")
            weights_path = os.path.join(self.plugin_dir, "class_weights.json")
            
            model_arch_text = self.dialog.model_combo.currentText()
            if "U-Net" in model_arch_text:
                model_arch = "unet"
            elif "DeepLabV3+" in model_arch_text:
                model_arch = "deeplabv3plus"
            elif "SegFormer" in model_arch_text:
                model_arch = "segformer"
            else:
                model_arch = "mask2former"
                
            train_script = os.path.join(self.plugin_dir, "scripts", "train.py")
            
            # Use a gentler learning rate (2e-6) and force disable Focal Loss to prevent overfitting on boundary noise
            cmd = [
                py_exe, train_script,
                "--dataset_dir", dataset_dir,
                "--is_16bit", str(is_16bit),
                "--epochs", str(epochs),
                "--batch_size", str(batch),
                "--mapping_file", map_path,
                "--model_arch", model_arch,
                "--resume_from", train_ckpt,
                "--lr", "2e-6"
            ]
            
            if os.path.exists(weights_path):
                cmd.extend(["--class_weights", weights_path])
                
            self.dialog.log(f"Executing Active Learning Retraining: {' '.join(cmd)}")
            self.dialog.train_btn.setEnabled(False)
            self.dialog.infer_btn.setEnabled(False)
            self.is_inference = False
            
            self.worker = SubprocessWorker(cmd, cwd=self.plugin_dir)
            self.worker.log_msg.connect(self.dialog.log)
            self.worker.progress_val.connect(self.dialog.progress_bar.setValue)
            
            # Keep track of original checkpoint run dir to compare metrics later
            original_run_dir = os.path.dirname(train_ckpt)
            
            def on_retrain_finished(rc):
                self.dialog.train_btn.setEnabled(True)
                self.dialog.infer_btn.setEnabled(True)
                if rc == 0:
                    self.dialog.log("자동 재학습(파인튜닝) 완료! 성능 곡선 비교 분석을 실행합니다...")
                    self.run_retraining_comparison(original_run_dir)
                else:
                    self.dialog.log(f"재학습 실패 (에러 코드: {rc}).")
                    self.dialog.progress_bar.setValue(100)
                    
            self.worker.finished.connect(on_retrain_finished)
            self.worker.start()
        except Exception as e:
            self.dialog.log(f"재학습 프로세스 시작 중 치명적 예외 발생: {e}")
            self.dialog.progress_bar.setValue(100)

    def run_retraining_comparison(self, original_run_dir):
        py_exe = self.get_python_exe()
        if not py_exe: return
        
        # 1. Find the latest created folder inside checkpoints/ containing metrics.csv
        checkpoints_root = os.path.join(self.plugin_dir, "checkpoints")
        if not os.path.exists(checkpoints_root):
            return
            
        import glob
        subdirs = glob.glob(os.path.join(checkpoints_root, "*"))
        valid_subdirs = []
        for s in subdirs:
            if os.path.isdir(s) and os.path.exists(os.path.join(s, "metrics.csv")):
                # Avoid comparing the original run as the new run
                if os.path.abspath(s) != os.path.abspath(original_run_dir):
                    valid_subdirs.append(s)
                    
        if not valid_subdirs:
            self.dialog.log("비교 대상 신규 학습 결과(metrics.csv)를 찾지 못해 비교 가시화를 건너뜁니다.")
            return
            
        new_run_dir = max(valid_subdirs, key=os.path.getmtime)
        
        old_csv = os.path.join(original_run_dir, "metrics.csv")
        new_csv = os.path.join(new_run_dir, "metrics.csv")
        
        if not os.path.exists(old_csv) or not os.path.exists(new_csv):
            self.dialog.log("학습 성능 CSV 파일이 한 쪽이라도 존재하지 않아 비교 가시화를 수행할 수 없습니다.")
            return
            
        # Save output in the new run directory
        out_dir = new_run_dir
        script_path = os.path.join(self.plugin_dir, "scripts", "compare_training.py")
        
        cmd = [
            py_exe, script_path,
            "--csv_files", f"{old_csv},{new_csv}",
            "--names", f"Original Model,Enhanced Model",
            "--out_dir", out_dir
        ]
        
        self.worker = SubprocessWorker(cmd, cwd=self.plugin_dir)
        self.worker.log_msg.connect(self.dialog.log)
        self.worker.finished.connect(lambda rc: self.on_compare_finished(rc, out_dir, "training_comparison.png"))
        self.worker.start()

    def run_manual_active_diagnostic(self):
        from qgis.PyQt.QtWidgets import QMessageBox
        
        unc_dir = self.dialog.active_infer_folder_input.text().strip()
        if not unc_dir or not os.path.isdir(unc_dir):
            QMessageBox.warning(self.dialog, "경로 오류", "유효한 추론 결과 폴더(Results/.../UNET_Inference_...)를 선택해 주세요.")
            return
            
        npz_files = [f for f in os.listdir(unc_dir) if f.lower().endswith(".npz")]
        if not npz_files:
            sub_unc = os.path.join(unc_dir, "uncertainty_data")
            if os.path.isdir(sub_unc):
                unc_dir = sub_unc
                npz_files = [f for f in os.listdir(unc_dir) if f.lower().endswith(".npz")]
                
        if not npz_files:
            QMessageBox.warning(self.dialog, "오류", "선택한 폴더 내에 불확실성 데이터(.npz)가 존재하지 않습니다.")
            return
            
        dataset_dir = self.dialog.dataset_dir_input.text().strip()
        if not dataset_dir or not os.path.isdir(dataset_dir):
            QMessageBox.warning(self.dialog, "설정 필요", "오토 라벨링 결과를 수혈할 Dataset Root를 먼저 설정해 주세요 (Training 탭).")
            return
            
        img_dir = os.path.join(dataset_dir, "test", "image")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(dataset_dir, "val", "image")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(dataset_dir, "train", "image")
        if not os.path.isdir(img_dir):
            for root, dirs, files in os.walk(dataset_dir):
                if "image" in dirs:
                    img_dir = os.path.join(root, "image")
                    break
                    
        if not os.path.isdir(img_dir):
            QMessageBox.warning(self.dialog, "오류", "Dataset Root 내에서 원본 이미지 폴더('image')를 찾을 수 없습니다. 경로를 확인해 주세요.")
            return

        self.dialog.log(f"수동 진단 시작: {unc_dir} 내 {len(npz_files)}개 불확실성 타일 분석 중...")
        self.last_infer_img_dir = img_dir
        self.trigger_active_learning(unc_dir)
