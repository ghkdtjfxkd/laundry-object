from ultralytics import YOLO
import cv2
import numpy as np


# 1. YOLO 모델 로드
def load_model(model_path="yolo11n.pt"):
    """
    YOLO 모델을 로드합니다.
    :param model_path: 사전 학습된 모델 경로
    :return: YOLO 모델 객체
    """
    try:
        model = YOLO(model_path)
        print("YOLO 모델 로드 완료!")
        return model
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        exit()


# 2. 이미지에서 객체 탐지
def detect_and_display(image_path, model):
    """
    이미지를 입력받아 YOLO로 객체를 탐지하고 결과를 표시합니다.
    :param image_path: 입력 이미지 경로
    :param model: YOLO 모델 객체
    """
    try:
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 로드할 수 없습니다: {image_path}")
            return

        # YOLO 예측 실행
        results = model.predict(source=image, show=False)
        detections = results[0].boxes.data.cpu().numpy()

        # 탐지 결과 시각화
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            class_name = model.names[int(class_id)]
            print(f"Detected {class_name} with confidence {confidence:.2f}")

            # 경계 상자 그리기
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        # 탐지 결과 출력
        cv2.imshow("YOLO Detection Results", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"탐지 중 오류 발생: {e}")


# 3. 실행
if __name__ == "__main__":
    # 모델 로드
    model = load_model("yolo11n.pt")

    # 탐지할 이미지 경로
    image_path = "images.jpeg"  # 여기 이미지 경로를 실제 경로로 변경

    # 객체 탐지 및 시각화
    detect_and_display(image_path, model)
