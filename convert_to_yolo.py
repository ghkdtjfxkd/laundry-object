import os
from PIL import Image
import torchvision.transforms as transforms

def convert_to_yolo_format(dataset, output_dir="dataset"):
    """
    Fashion MNIST 데이터를 YOLO 형식으로 변환합니다.
    :param dataset: Fashion MNIST 데이터셋 객체
    :param output_dir: 변환된 데이터를 저장할 디렉토리
    """
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/train", exist_ok=True)

    for i, (image, label) in enumerate(dataset):
        # 이미지 저장
        img_path = f"{output_dir}/images/train/image_{i}.jpg"
        pil_image = transforms.ToPILImage()(image)
        pil_image.save(img_path)

        # 라벨 저장 (YOLO 형식)
        label_path = f"{output_dir}/labels/train/image_{i}.txt"
        with open(label_path, "w") as f:
            f.write(f"{label} 0.5 0.5 1.0 1.0\n")  # 단순 중심 좌표와 크기 지정

    print(f"YOLO 형식 데이터가 '{output_dir}' 디렉토리에 저장되었습니다.")
