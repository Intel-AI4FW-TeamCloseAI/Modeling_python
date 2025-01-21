import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
import onnxruntime as ort

# ------------------------------
# 1) 환경 설정
# ------------------------------

# 폴더 디렉토리 구조
# 현재 폴더
# ├── final_onnx.py (현재 파일)
# ├── yolov11n-face.onnx
# ├── deepfake_binary_s128_e5_early.onnx
# └── image.jpg

# base_dir = os.getcwd()   # 현재 작업 디렉토리 (.ipynb 파일과 같은 경로)
base_dir = os.path.dirname(os.path.abspath(__file__))   # 현재 작업 디렉토리 (.py 파일과 같은 경로)

# YOLOv8 얼굴 검출 모델(.onnx) 경로
face_model_path = os.path.join(base_dir, "yolov11n-face.onnx")

# EfficientNet 분류 모델(.onnx) 경로
cls_model_path = os.path.join(base_dir, "deepfake_binary_s128_e5_early.onnx")

# 예측 대상 이미지
test_image_path = os.path.join(base_dir, "fake0006.jpg")

# ------------------------------
# 2) 모델 로드
# ------------------------------
# (a) YOLOv11 얼굴 검출 모델 로드 (onnx)
face_detector = YOLO(face_model_path)

# (b) EfficientNet 분류 모델 로드 (ONNX Runtime 사용)
# ----------------------------------------------
#  1) onnxruntime 세션 생성
# ----------------------------------------------
cls_session = ort.InferenceSession(cls_model_path, providers=["CPUExecutionProvider"])
# GPU용 onnxruntime 사용 시, 설치 후 providers=["CUDAExecutionProvider"] 등 설정 가능

# 입력/출력 노드 이름(옵션)
input_name = cls_session.get_inputs()[0].name
output_name = cls_session.get_outputs()[0].name
print(f"[INFO] ONNX Model input_name = {input_name}, output_name = {output_name}")

# 분류 모델 전처리: (128, 128)로 resize 후 Normalize
img_resized = (128, 128)
transform = transforms.Compose([
    transforms.Resize(img_resized),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 클래스 인덱스 → 문자열 매핑 (학습 순서에 맞게)
idx_to_class = {0: "Fake", 1: "Real"}

# ------------------------------
# 3) 얼굴 검출 + Crop + 분류 함수
# ------------------------------
def detect_and_classify_faces(image_path, detector, onnx_sess, extend_ratio=0.5):
    """
    1. YOLO로 얼굴 검출
    2. 얼굴 박스를 (extend_ratio) 비율만큼 상하좌우 확장
    3. Crop 후 ONNX 모델(EfficientNet) 예측
    4. 얼굴이 하나도 없는 경우 -> (선택) 전체 이미지로 분류 + 안내 메시지
    """
    # 원본 이미지 로드
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[Error] Cannot read image from {image_path}")
        return
    
    height, width, _ = img_bgr.shape

    # YOLOv11n-face 추론
    results = detector.predict(source=img_bgr, conf=0.8)
    
    detected_any_face = False

    for r in results:
        boxes = r.boxes
        if len(boxes) == 0:
            continue  # 해당 result에 박스가 전혀 없으면 다음 result 확인

        for box in boxes:
            detected_any_face = True

            x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # 바운딩박스 좌표
            conf = box.conf[0].item()

            # ------------------------------
            # (a) 얼굴 박스 확장
            # ------------------------------
            face_width = x2 - x1
            face_height = y2 - y1

            # 상단, 하단 확장
            new_y1 = max(0, y1 - int(face_height * extend_ratio * 0.7))  
            new_y2 = min(height, y2 + int(face_height * extend_ratio))

            # 좌우 확장
            extend_w = int(face_width * extend_ratio * 0.7)
            new_x1 = max(0, x1 - extend_w)
            new_x2 = min(width, x2 + extend_w)

            # Crop 영역 (BGR)
            crop_bgr = img_bgr[new_y1:new_y2, new_x1:new_x2]

            # ------------------------------
            # (b) ONNX 분류 모델로 예측
            # ------------------------------
            # 1) PIL 변환
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)

            # 2) transform 적용 (torch.Tensor)
            input_tensor = transform(pil_img).unsqueeze(0)  # shape: (1, 3, 128, 128)
            
            # 3) onnxruntime에 맞게 numpy 변환
            input_data = input_tensor.cpu().numpy()  # float32 형태

            # 4) 모델 추론
            outputs = onnx_sess.run(None, {input_name: input_data})
            # outputs[0]의 shape가 (1, 3)이라 가정 -> argmax로 클래스 인덱스 추출
            pred_np = outputs[0]  # 첫 번째 Output
            label_idx = int(np.argmax(pred_np, axis=1)[0])

            # (c) 예측 라벨
            label_str = idx_to_class.get(label_idx, "Unknown")
            print(f" - 얼굴 분류 결과: {label_str}")

            # ------------------------------
            # (d) 원본 이미지에 결과 표시
            # ------------------------------
            color = (0, 255, 0)  # (B, G, R)
            cv2.rectangle(img_bgr, (new_x1, new_y1), (new_x2, new_y2), color, 2)

            # 텍스트: 분류 결과 + YOLO 검출 confidence
            text_str = f"{label_str}({conf:.2f})"
            cv2.putText(img_bgr, text_str, (new_x1, new_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # ------------------------------
    # 4) 얼굴 미검출 시 처리
    # ------------------------------
    if not detected_any_face:
        print("[Info] 얼굴을 감지하지 못했습니다.")

        # (선택) 원본 전체 이미지를 분류 모델에 넣어볼 수도 있음
        crop_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        input_tensor = transform(pil_img).unsqueeze(0)
        input_data = input_tensor.cpu().numpy()

        outputs = onnx_sess.run(None, {input_name: input_data})
        pred_np = outputs[0]
        label_idx = int(np.argmax(pred_np, axis=1)[0])
        label_str = idx_to_class.get(label_idx, "Unknown")

        print(f" - 전체 이미지 분류 결과: {label_str}")

        # (선택) 원본 이미지에 텍스트 표시
        cv2.putText(img_bgr, f"{label_str} (No Face)", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # 최종 결과 이미지 표시
    cv2.imshow("Face + Classification", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------------------------------
# 4) 메인 실행
# ------------------------------
if __name__ == "__main__":
    detect_and_classify_faces(
        image_path=test_image_path,
        detector=face_detector,
        onnx_sess=cls_session,
        extend_ratio=0.5  # 높이/너비 확장 비율
    )
