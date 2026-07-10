# -*- coding: utf-8 -*-
"""
오프라인 폐쇄망 설치를 위한 종속성 및 AI 모델 사전 다운로드 스크립트.
이 스크립트는 **반드시 인터넷이 연결된 PC**에서 먼저 실행되어야 합니다.
실행이 완료되면 offline_setup 폴더 안에 wheels(설치 파일)과 models(가중치)가 저장됩니다.
"""

import os
import sys
import subprocess

def download_wheels():
    print("1. 패키지 설치 파일(.whl) 다운로드 시작...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wheels_dir = os.path.join(base_dir, 'offline_setup', 'wheels')
    req_file = os.path.join(base_dir, 'requirements.txt')
    
    if not os.path.exists(wheels_dir):
        os.makedirs(wheels_dir)
        
    if not os.path.exists(req_file):
        print(f"오류: {req_file} 파일을 찾을 수 없습니다.")
        return False
        
    try:
        # pip download를 사용하여 종속성을 모두 다운로드
        subprocess.check_call([
            sys.executable, "-m", "pip", "download", 
            "-d", wheels_dir, 
            "-r", req_file
        ])
        print("✅ 패키지 다운로드 완료.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 패키지 다운로드 실패: {e}")
        return False

def download_models():
    print("2. Mask2Former AI 모델 가중치 다운로드 시작...")
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub가 없습니다. 설치합니다...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download
        
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    models = {
        "mask2former-swin-tiny": "facebook/mask2former-swin-tiny-cityscapes-semantic",
        "mask2former-swin-large": "facebook/mask2former-swin-large-cityscapes-semantic"
    }
    
    success = True
    for folder, repo_id in models.items():
        models_dir = os.path.join(base_dir, 'offline_setup', 'models', folder)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        print(f"Downloading {repo_id} to {models_dir}...")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
            print(f"✅ {folder} 다운로드 완료.")
        except Exception as e:
            print(f"❌ {folder} 다운로드 실패: {e}")
            success = False
    return success

if __name__ == "__main__":
    print("=" * 50)
    print("폐쇄망용 데이터 사전 다운로드 스크립트")
    print("=" * 50)
    
    success_wheels = download_wheels()
    success_models = download_models()
    
    if success_wheels and success_models:
        print("\n모든 준비가 완료되었습니다.")
        print("이제 플러그인 폴더 전체를 폐쇄망 PC로 이동하여 QGIS에 설치하세요.")
    else:
        print("\n다운로드 중 오류가 발생했습니다. 인터넷 연결 및 권한을 확인하세요.")
