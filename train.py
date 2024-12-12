from ultralytics import YOLO

def train_model(data_yaml="dataset.yaml", model_name="yolo11n.pt", epochs=5, imgsz=640):
    """
    YOLO 모델을 학습시킵니다.
    :param data_yaml: YOLO 학습 데이터 구성 파일 경로
    :param model_name: YOLO 사전 학습된 모델 경로
    :param epochs: 학습 epoch 수
    :param imgsz: 학습 이미지 크기
    """
    # YOLO 모델 로드
    model = YOLO(model_name)

    # 모델 학습
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)

    print("YOLO 모델 학습 완료!")

if __name__ == "__main__":
    # YOLO 모델 학습
    train_model(data_yaml="dataset.yaml", model_name="yolo11n.pt", epochs=5)
