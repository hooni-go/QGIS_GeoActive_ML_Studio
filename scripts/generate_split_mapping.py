import os
import glob
import csv
import re
from collections import defaultdict

def main():
    base_dir = r"D:\data\LC_Air_GS-JL-GG_sp2_GeoSplit"
    out_csv = r"D:\data\LC_Air_GS-JL-GG_sp2_GeoSplit\split_mapping.csv"
    
    print("GeoSplit 폴더 구조 스캔 중...")
    
    # 맵핑 정보를 담을 딕셔너리
    sheet_info = defaultdict(lambda: {"Region": "", "Split": set(), "Count": 0})
    pattern = re.compile(r"LC_([A-Z]{2})_[^_]+_(\d+)_")
    
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(base_dir, split, "image")
        if not os.path.exists(img_dir):
            continue
            
        images = glob.glob(os.path.join(img_dir, "*.tif"))
        for img_path in images:
            basename = os.path.basename(img_path)
            match = pattern.search(basename)
            if match:
                region = match.group(1)
                map_sheet = match.group(2)
                
                sheet_info[map_sheet]["Region"] = region
                sheet_info[map_sheet]["Split"].add(split.capitalize())
                sheet_info[map_sheet]["Count"] += 1
                
    print(f"총 {len(sheet_info)}개의 도엽(Map Sheet) 블록 분석 완료.")
    
    # CSV 저장 (한글 깨짐 방지를 위해 utf-8-sig 사용)
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # QGIS Join 시 헷갈리지 않도록 명확한 영문 헤더 사용
        writer.writerow(["MapSheet", "Region", "Split", "ImageCount"])
        
        leak_count = 0
        for sheet, info in sheet_info.items():
            split_list = sorted(list(info["Split"]))
            split_str = "+".join(split_list)
            
            # Data Leakage 점검
            if len(split_list) > 1:
                print(f"[경고] Data Leakage 발생! 도엽 {sheet}가 {split_str} 에 섞여 있습니다.")
                leak_count += 1
                
            writer.writerow([sheet, info["Region"], split_str, info["Count"]])
            
    if leak_count == 0:
        print("훌륭합니다! 교차 분할(Data Leakage)이 0건으로 완벽하게 격리되었습니다.")
        
    print(f"\n매핑 테이블 생성이 완료되었습니다!")
    print(f"저장 경로: {out_csv}")

if __name__ == "__main__":
    main()
