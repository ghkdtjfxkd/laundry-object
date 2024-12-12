import cv2
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

# YOLO 모델 경로 설정
model_path = "yolo11n.pt"

# 디바이스 설정 (GPU 사용 가능 시 'cuda')
device = select_device('cpu')

# YOLO 모델 로드
model = DetectMultiBackend(model_path, device=device)

# 클래스 이름 정의 (예: 0: 배경, 1: 상의, 2: 하의)
class_names = ['background', 'shirt', 'pants']


def detect_objects():
    # 이미지 파일 경로
    image_path = "images.jpeg"  # 같은 패키지 내에 있는 이미지 파일 사용

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 로드할 수 없습니다. 파일 경로를 확인하세요.")
        return

    orig_image = image.copy()

    # YOLO 입력 크기로 변환
    img_size = 640
    image_resized = cv2.resize(image, (img_size, img_size))
    image_resized = image_resized.transpose((2, 0, 1))  # 채널 변환 (HWC -> CHW)
    image_resized = torch.from_numpy(image_resized).float() / 255.0  # NumPy 배열을 PyTorch 텐서로 변환
    image_resized = image_resized.unsqueeze(0)  # 배치 차원 추가 (1, C, H, W)

    # 모델 추론
    results = model(image_resized)  # YOLO 모델 추론
    detections = non_max_suppression(results, conf_thres=0.25, iou_thres=0.45)  # NMS 적용

    # 바운딩 박스 그리기
    for det in detections:
        if det is not None:
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)  # 바운딩 박스 좌표
                label = f"{class_names[int(cls)]} {conf:.2f}"

                # 바운딩 박스와 라벨 표시
                cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(orig_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 출력
    cv2.imshow("YOLO Detection", orig_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_objects()
