import os
import glob
import shutil
import re
from collections import defaultdict
import random
import time

def main():
    src_dir = r"D:\data\LC_Air_GS-JL-GG_sp2"
    dst_dir = r"D:\data\LC_Air_GS-JL-GG_sp2_GeoSplit"
    
    print("============================================")
    print("   Spatial Block Cross-Validation Split")
    print("============================================")
    
    print("1. 파일 스캔 중 (image 및 label 폴더 매핑)...")
    all_images = glob.glob(os.path.join(src_dir, "**", "image", "*.tif"), recursive=True)
    
    # 추론 결과 폴더 등 불필요한 폴더 배제 (오직 train, val, test 폴더 내의 image만)
    all_images = [f for f in all_images if "\\train\\" in f or "\\val\\" in f or "\\test\\" in f]
    
    # 파일명 예시: LC_GG_AP25_36701004_015_2021.tif
    # 매칭: 그룹1 = 지역(GG), 그룹2 = 도엽번호(36701004)
    pattern = re.compile(r"LC_([A-Z]{2})_[^_]+_(\d+)_")
    
    # region -> map_sheet -> list of file paths
    region_map = defaultdict(lambda: defaultdict(list))
    
    valid_count = 0
    for img_path in all_images:
        basename = os.path.basename(img_path)
        match = pattern.search(basename)
        if match:
            region = match.group(1)
            map_sheet = match.group(2)
            region_map[region][map_sheet].append(img_path)
            valid_count += 1
        else:
            print(f"Warning: Could not parse {basename}")
            
    print(f"총 {valid_count} 쌍의 유효한 위성 영상을 스캔했습니다.")
    
    # 난수 시드 고정 (재현성 확보)
    random.seed(42)
    
    # 분할 비율: Train 75%, Test 15%, Val 10%
    split_ratios = {"train": 0.75, "val": 0.10, "test": 0.15}
    allocations = defaultdict(list)  # subset -> [img_paths]
    
    print("\n2. 권역별 도엽(Map Sheet) 기반 거대 블록(Macro) 분할 중...")
    for region, sheets in region_map.items():
        sheet_ids = list(sheets.keys())
        sheet_ids.sort() # 공간적 인접성 유지를 위해 도엽 번호 순으로 정렬 후 자름 (셔플 없음)
        
        n_total = len(sheet_ids)
        n_train = int(n_total * split_ratios["train"])
        n_test = int(n_total * split_ratios["test"])
        
        train_sheets = sheet_ids[:n_train]
        test_sheets = sheet_ids[n_train:n_train+n_test]
        val_sheets = sheet_ids[n_train+n_test:]
        
        # 합계 계산
        tr_cnt = sum(len(sheets[s]) for s in train_sheets)
        te_cnt = sum(len(sheets[s]) for s in test_sheets)
        va_cnt = sum(len(sheets[s]) for s in val_sheets)
        
        print(f"  [{region} 권역] 도엽 {n_total}개 -> Train: {len(train_sheets)}개({tr_cnt}장), Test: {len(test_sheets)}개({te_cnt}장), Val: {len(val_sheets)}개({va_cnt}장)")
        
        for s in train_sheets: allocations["train"].extend(sheets[s])
        for s in test_sheets: allocations["test"].extend(sheets[s])
        for s in val_sheets: allocations["val"].extend(sheets[s])
        
    print("\n3. 안전한 데이터 복사 (GeoSplit 새 폴더 생성)...")
    
    for subset in ["train", "val", "test"]:
        os.makedirs(os.path.join(dst_dir, subset, "image"), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, subset, "label"), exist_ok=True)
        
    def get_label_path(img_path):
        return img_path.replace("\\image\\", "\\label\\")
        
    # 복사 진행 (시간이 걸리므로 10%마다 로깅)
    total_files = sum(len(lst) for lst in allocations.values())
    copied = 0
    
    for subset, files in allocations.items():
        print(f"  {subset} 폴더로 {len(files)} 셋트 복사 시작...")
        for img_path in files:
            lbl_path = get_label_path(img_path)
            
            dst_img = os.path.join(dst_dir, subset, "image", os.path.basename(img_path))
            dst_lbl = os.path.join(dst_dir, subset, "label", os.path.basename(lbl_path))
            
            shutil.copy2(img_path, dst_img)
            if os.path.exists(lbl_path):
                shutil.copy2(lbl_path, dst_lbl)
                
            copied += 1
            if copied % 3000 == 0:
                print(f"    ... {copied}/{total_files} 완료")
                
    print("\n============================================")
    print(" 지리적 분할(Spatial Block Split) 완료!")
    print(f" 저장 경로: {dst_dir}")
    print(f" - Train : {len(allocations['train'])} 셋트")
    print(f" - Val   : {len(allocations['val'])} 셋트")
    print(f" - Test  : {len(allocations['test'])} 셋트")
    print("============================================")

if __name__ == "__main__":
    main()
