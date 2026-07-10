# -*- coding: utf-8 -*-
"""
오프라인 폐쇄망 무설치 실행을 위한 포터블 파이썬 환경 원클릭 구축 스크립트.
인터넷이 되는 PC에서 이 스크립트를 한 번만 실행하면, 플러그인 폴더 안에 'python_env'가 만들어집니다.
완성된 플러그인 폴더를 USB로 복사하여 폐쇄망 PC에 넣으면 아무 설치 과정 없이 즉시 실행됩니다.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil

PYTHON_VER = "3.10.11"
PYTHON_URL = f"https://www.python.org/ftp/python/{PYTHON_VER}/python-{PYTHON_VER}-embed-amd64.zip"

def build_portable_python(env_dir):
    print(f"1. 포터블 파이썬({PYTHON_VER}) 다운로드 및 압축 해제 중...")
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
        
    zip_path = os.path.join(env_dir, "python_embed.zip")
    if not os.path.exists(os.path.join(env_dir, "python.exe")):
        urllib.request.urlretrieve(PYTHON_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(env_dir)
        os.remove(zip_path)
        
    # pip 사용을 위해 python310._pth 파일 수정 (import site 주석 해제)
    pth_file = os.path.join(env_dir, "python310._pth")
    if os.path.exists(pth_file):
        with open(pth_file, 'r') as f:
            lines = f.readlines()
        with open(pth_file, 'w') as f:
            for line in lines:
                if line.strip() == "#import site":
                    f.write("import site\n")
                else:
                    f.write(line)
                    
    print("✅ 포터블 파이썬 기본 환경 구성 완료.")
    return os.path.join(env_dir, "python.exe")

def install_pip_and_packages(py_exe, base_dir):
    print("2. pip 및 딥러닝 패키지 설치 중... (시간이 오래 걸릴 수 있습니다)")
    
    # get-pip.py 다운로드
    env_dir = os.path.dirname(py_exe)
    get_pip_path = os.path.join(env_dir, "get-pip.py")
    if not os.path.exists(os.path.join(env_dir, "Scripts", "pip.exe")):
        urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip_path)
        subprocess.check_call([py_exe, get_pip_path])
        os.remove(get_pip_path)
        
    # requirements.txt 설치
    req_file = os.path.join(base_dir, 'requirements.txt')
    if os.path.exists(req_file):
        try:
            subprocess.check_call([py_exe, "-m", "pip", "install", "-r", req_file])
            print("✅ 딥러닝 패키지 설치 완료.")
        except subprocess.CalledProcessError as e:
            print(f"❌ 패키지 설치 실패: {e}")
            return False
    else:
        print("경고: requirements.txt 파일이 없습니다.")
    return True

def download_models(py_exe, base_dir):
    print("3. Mask2Former AI 모델 가중치 다운로드 시작...")
    try:
        subprocess.check_call([py_exe, "-m", "pip", "install", "huggingface_hub"])
    except Exception:
        pass
        
    models_dir = os.path.join(base_dir, 'offline_setup', 'models', 'mask2former-swin-tiny')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    # Python 스크립트를 통해 huggingface 모델 다운로드
    script = f"""
from huggingface_hub import snapshot_download
snapshot_download(repo_id='facebook/mask2former-swin-tiny-cityscapes-semantic', local_dir=r'{models_dir}', local_dir_use_symlinks=False)
"""
    try:
        subprocess.check_call([py_exe, "-c", script])
        print("✅ 모델 다운로드 완료.")
        return True
    except Exception as e:
        print(f"❌ 모델 다운로드 실패: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print(" 폐쇄망용 완벽 포터블 파이썬 환경(Zero-Install) 구축기 ")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_dir = os.path.join(base_dir, "python_env")
    
    py_exe = build_portable_python(env_dir)
    success_pkgs = install_pip_and_packages(py_exe, base_dir)
    success_models = download_models(py_exe, base_dir)
    
    if success_pkgs and success_models:
        print("\n" + "=" * 60)
        print("🎉 [구축 완료] 모든 세팅이 완료되었습니다.")
        print(f"이제 '{os.path.basename(base_dir)}' 폴더 전체를 복사하여 폐쇄망 PC에 가져가세요.")
        print("폐쇄망에서는 QGIS 플러그인만 실행하면 어떠한 버튼 클릭이나 추가 설치 없이 바로 작동합니다!")
        print("=" * 60)
    else:
        print("\n오류가 발생했습니다. 인터넷 환경 및 권한을 확인해 주세요.")
