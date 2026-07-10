# -*- coding: utf-8 -*-
"""
Utility functions for QGIS_GeoActive ML Studio Plugin.
Includes dependency checking, reproducibility control, and LaTeX table generation.
"""

import os
import sys
import subprocess
import importlib

def check_and_install_dependencies():
    """
    폐쇄망(오프라인) 환경에서 필수 라이브러리가 설치되어 있는지 확인하고, 
    없으면 플러그인 내부에 준비된 wheels를 통해 로컬 설치를 진행합니다.
    """
    required_packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('openpyxl', 'openpyxl'),
        ('numpy', 'numpy'),
        ('PIL', 'Pillow')
    ]
    
    missing_packages = []
    for module_name, package_name in required_packages:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_packages.append(package_name)
            
    if missing_packages:
        print(f"Missing packages detected: {missing_packages}. Attempting offline installation...")
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        wheels_dir = os.path.join(base_dir, 'offline_setup', 'wheels')
        req_file = os.path.join(base_dir, 'requirements.txt')
        
        if not os.path.exists(wheels_dir) or not os.path.exists(req_file):
            raise Exception("오프라인 설치 파일(wheels)이나 requirements.txt가 없습니다. 인터넷이 연결된 PC에서 download_dependencies.py를 먼저 실행하세요.")
            
        try:
            # sys.executable in QGIS points to qgis-bin.exe, which causes a new QGIS window to open.
            # Instead, we use sys.exec_prefix to find the actual python.exe.
            if sys.platform == 'win32':
                python_executable = os.path.join(sys.exec_prefix, 'python.exe')
            else:
                python_executable = os.path.join(sys.exec_prefix, 'bin', 'python')
                
            if not os.path.exists(python_executable):
                python_executable = "python" # Fallback to PATH
                
            # 오프라인 설치 명령어: 인터넷 연결 무시(--no-index) 및 로컬 폴더 탐색(--find-links)
            subprocess.check_call([
                python_executable, "-m", "pip", "install", 
                "--user", 
                "--no-index", 
                f"--find-links={wheels_dir}", 
                "-r", req_file
            ])
            print("Successfully installed missing packages offline.")
            
            # 설치 직후 import를 위해 sys.path 갱신
            import site
            importlib.invalidate_caches()
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to install offline packages: {e}")
            raise Exception(f"필수 패키지 오프라인 설치 실패. 에러: {e}")

def set_reproducibility(seed: int):
    """
    논문(Ablation Study)을 위한 완벽한 재현성을 보장하기 위해 난수 시드를 고정합니다.
    """
    try:
        import torch
        import numpy as np
        import random
        import os

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 멀티 GPU 환경
        
        # cuDNN 알고리즘을 결정론적으로 설정 (학술적 엄밀성)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 추가적인 PyTorch 결정론적 연산 보장
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass # 의존성 설치 전 호출 방지

def clear_vram():
    """
    GPU 메모리 누수를 방지하기 위해 각 Run/Epoch 종료 시 VRAM을 초기화합니다.
    """
    try:
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    except ImportError:
        pass

def generate_latex_table(df) -> str:
    """
    Pandas DataFrame을 학술지 양식에 맞는 LaTeX 표 코드로 변환합니다.
    """
    try:
        # 간단한 스타일링 적용 (booktabs 형식)
        latex_code = df.to_latex(index=False, escape=False, 
                                 column_format='|'.join(['c'] * len(df.columns)))
        
        # Booktabs 스타일 수동 적용 (to_latex 파라미터가 버전에 따라 다르므로 수동 치환)
        latex_code = latex_code.replace('\\toprule', '\\hline\\hline')
        latex_code = latex_code.replace('\\midrule', '\\hline')
        latex_code = latex_code.replace('\\bottomrule', '\\hline\\hline')
        
        return latex_code
    except Exception as e:
        return f"% LaTeX generation failed: {e}"
