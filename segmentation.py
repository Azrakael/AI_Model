#%%
from PIL import Image
import os
import csv

# 경로 설정
base_dir = './Image Dataset/Images'
preprocessed_dir = './processed_images'
labels_csv_path = './labels.csv'

# 제품 ID에 대한 하위 디렉토리 이름 설정
sub_dirs = {
    '20001': 'Cap Off Pear Off, PAD topside muscle',
    '20002': 'Cap off, pear on Topside muscle',
    '20003': 'Topside Heart muscle',
    '20004': 'Topside Bullet muscle',
    '20010': 'Cap Off, Non-Pad, Blue Skin Only Topside muscle'
}

# 날짜 폴더 리스트
date_folders = [folder for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]

# 라벨 파일 생성
with open(labels_csv_path, 'w', newline='', encoding='utf-8-sig') as labels_file:
    labels_writer = csv.writer(labels_file)
    labels_writer.writerow(['Image Name', 'Label'])

    # 각 날짜 폴더에 대해 반복
    for date_folder in date_folders:
        date_folder_path = os.path.join(base_dir, date_folder)
        
        # 날짜 폴더 내의 모든 이미지 파일에 대해 반복
        for img_name in os.listdir(date_folder_path):
            # 파일명 구조를 기반으로 식별 정보 추출
            if img_name.startswith('camera-screenshot'):
                plant_id = img_name[17:20]  # 식물 ID는 17~19번째 문자
                product_id = img_name[20:25]  # 제품 ID는 20~24번째 문자
                timestamp = img_name[25:]  # 타임스탬프는 나머지 문자

                # 해당 제품 ID가 sub_dirs에 있는지 확인하고 레이블을 얻음
                if product_id in sub_dirs:
                    product_label = sub_dirs[product_id]
                    
                    # 제품 레이블 이름으로 폴더 생성
                    product_dir = os.path.join(preprocessed_dir, product_label)
                    if not os.path.exists(product_dir):
                        os.makedirs(product_dir)

                    # 이미지 처리
                    img_path = os.path.join(date_folder_path, img_name)
                    save_name = img_name
                    save_path = os.path.join(product_dir, save_name)
                    
                    # 이미지 리사이징 및 저장
                    with Image.open(img_path) as img:
                        img_resized = img.resize((224, 224))  # VGG16 모델에 맞는 크기
                        img_resized.save(save_path, 'JPEG')  # JPG 형식으로 저장

                    # 라벨 정보를 CSV 파일에 기록
                    labels_writer.writerow([save_name, product_label])

# %%
