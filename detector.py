import os
from ultralytics import YOLO
import cv2
import random
from collections import Counter

# YOLOv8 모델 로드
model = YOLO('best.pt')

class_colors = {
    'leaf': (255, 0, 0),  # 파란색 
    'diseased_leaf': (0, 0, 255),  # 빨간색
}

# 나머지 클래스에 대해 임의 색상 생성
num_classes = len(model.names)
for cls_id, cls_name in enumerate(model.names):
    if cls_name not in class_colors:  # 이미 지정되지 않은 클래스만
        class_colors[cls_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# 폴더 내 모든 이미지 파일 가져오기
input_folder = 'chocomint'  # 이미지 파일이 있는 폴더
output_folder = 'chocomint_result'  # 결과 파일을 저장할 폴더
os.makedirs(output_folder, exist_ok=True) 

image_extensions = ['.png', '.jpg', '.jpeg']  # 이미지 확장자
image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in image_extensions]

for image_path in image_paths:
    # 추론 수행 (conf 값 설정)
    conf_threshold = 0.25  # confidence 값
    results = model(image_path, conf=conf_threshold)

    # 클래스별 객체 개수를 세기 위한 Counter
    class_counts = Counter()

    # 결과를 반복하며 박스와 클래스 이름 그리기
    for result in results:
        img = result.orig_img  # 원본 이미지
        for box in result.boxes.data:  # 박스 정보
            x1, y1, x2, y2 = map(int, box[:4])  # 좌표 추출
            cls = int(box[5])  # 클래스 ID
            label = model.names[cls]  # 클래스 이름
            color = class_colors.get(label, (0, 255, 0))  # 클래스 이름 기반 색상 가져오기
            
            # 클래스 개수 업데이트
            class_counts[label] += 1

            # 박스 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # 클래스 이름 텍스트 추가
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 클래스별 개수를 왼쪽 위에 추가
        fixed_order = ['leaf', 'diseased_leaf']
        y_offset = 40
        for cls_name in fixed_order:
            if cls_name in class_counts:
                count = class_counts[cls_name]
                label = f"{cls_name}: {count}"
                color = class_colors.get(cls_name, (0, 255, 0))
                cv2.putText(img, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)  # 폰트 조절: fontScale=1.0, thickness=2
                y_offset += 40  # 다음 줄로 이동

        # 추후 기타 클래스 추가 (고정 순서에 없는 경우)
        for cls_name, count in class_counts.items():
            if cls_name not in fixed_order:
                label = f"{cls_name}: {count}"
                color = class_colors.get(cls_name, (0, 255, 0))
                cv2.putText(img, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                y_offset += 40  # 다음 줄로 이동
                
                
         # leaf 와 diseased_leaf 의 비율 계산
        leaf_count = class_counts.get('leaf', 0)
        diseased_count = class_counts.get('diseased_leaf', 0)
        total_count = leaf_count + diseased_count

        if total_count > 0:
            leaf_ratio = (leaf_count / total_count) * 100  # leaf 비율 계산
        else:
            leaf_ratio = 0

        # 비율 여부에 따라 정상 메시지 설정
        if leaf_ratio >= 80:
            status = 'Normal'
            color = (0, 255, 0)  # 정상
        elif 60 <= leaf_ratio < 80:
            status = 'Caution'
            color = (0, 255, 255)  # 주의
        elif 40 <= leaf_ratio < 60:
            status = 'Warning'
            color = (0, 165, 255)  # 경고
        elif 20 <= leaf_ratio < 40:
            status = 'Danger'
            color = (0, 0, 255)  # 위험
        else:
            status = 'Critical'
            color = (128, 0, 128)  # 폐사

        # 상황 메시지를 우측 상단에 표시
        status_text = f"Status: {status} ({leaf_ratio:.1f}%)"
        cv2.putText(img, status_text, (img.shape[1] - 400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # 결과 이미지 저장
        output_path = os.path.join(output_folder, os.path.basename(image_path).replace('.png', '_output.jpg').replace('.jpg', '_output.jpg'))
        cv2.imwrite(output_path, img)
        print(f"Processed image saved to {output_path}")
