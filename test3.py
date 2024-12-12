from download_fashion_mnist import download_fashion_mnist
from convert_to_yolo import convert_to_yolo_format

if __name__ == "__main__":
    # 데이터 다운로드
    train_dataset, _ = download_fashion_mnist()

    # YOLO 형식 변환
    convert_to_yolo_format(train_dataset)
