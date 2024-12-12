import cv2
from ultralytics import YOLO

def detect_objects(image_path, model_path="runs/train/weights/best.pt"):
    """
    학습된 YOLO 모델로 객체를 탐지합니다.
    :param image_path: 입력 이미지 경로
    :param model_path: 학습된 YOLO 모델 경로
    """
    # 학습된 YOLO 모델 로드
    model = YOLO(model_path)

    # 입력 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지 로드 실패: {image_path}")
        return

    # YOLO 탐지
    results = model.predict(source=image, show=False)

    # 탐지 결과 시각화
    detections = results[0].boxes.data.cpu().numpy()
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        class_name = model.names[int(class_id)]
        print(f"Detected {class_name} with confidence {confidence:.2f}")
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # 탐지 결과를 로컬 창에 표시
    cv2.imshow("YOLO Detection Results", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 실행 코드
if __name__ == "__main__":
    # 탐지할 이미지 경로
    test_image_path = "datasets/images/train/image_0.jpg"

    # 탐지 실행
    detect_objects(image_path=test_image_path)
