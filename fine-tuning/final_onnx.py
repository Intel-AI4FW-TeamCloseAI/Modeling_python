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

# base_dir = os.getcwd()   # 현재 작업 디렉토리 (.ipynb 파일과 같은 경로)
base_dir = os.path.dirname(os.path.abspath(__file__))   # 현재 작업 디렉토리 (.py 파일과 같은 경로)

# YOLOv8 얼굴 검출 모델(.onnx) 경로
face_model_path = os.path.join(base_dir, "yolov11n-face.onnx")

# EfficientNet 분류 모델(.onnx) 경로
cls_model_path = os.path.join(base_dir, "deepfake_binary_s128_e5_early.onnx")

# 예측 대상 이미지 폴더
image_folder_path = os.path.join(base_dir, "sample")

# ------------------------------
# 2) 모델 로드
# ------------------------------
# (a) YOLOv11 얼굴 검출 모델 로드 (onnx)
face_detector = YOLO(face_model_path)

# (b) EfficientNet 분류 모델 로드 (ONNX Runtime 사용)
cls_session = ort.InferenceSession(cls_model_path, providers=["CPUExecutionProvider"])

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
def detect_and_classify_faces(image_path, detector, onnx_sess, extend_ratio=0.4):
    results_log = []  # 결과 로그 저장 리스트
    
    # 원본 이미지 로드
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[Error] Cannot read image from {image_path}")
        return None, None

    height, width, _ = img_bgr.shape

    # YOLOv11n-face 추론
    results = detector.predict(source=img_bgr, conf=0.8)
    
    detected_any_face = False

    for r in results:
        boxes = r.boxes
        if len(boxes) == 0:
            continue

        for box in boxes:
            detected_any_face = True

            x1, y1, x2, y2 = box.xyxy[0].int().tolist()

            # (a) 얼굴 박스 확장
            face_width = x2 - x1
            face_height = y2 - y1

            new_y1 = max(0, y1 - int(face_height * extend_ratio))  
            new_y2 = min(height, y2 + int(face_height * extend_ratio * 1.2))

            extend_w = int(face_width * extend_ratio)
            new_x1 = max(0, x1 - extend_w)
            new_x2 = min(width, x2 + extend_w)

            crop_bgr = img_bgr[new_y1:new_y2, new_x1:new_x2]

            # (b) ONNX 분류 모델로 예측
            # 전체 이미지를 RGB로 변환
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            
            # 전처리 수행
            input_tensor = transform(pil_img).unsqueeze(0)
            input_data = input_tensor.cpu().numpy()
            
            # ONNX 모델 추론
            outputs = onnx_sess.run(None, {input_name: input_data})
            pred_np = outputs[0]
            
            # 소프트맥스 적용
            softmax_output = np.exp(pred_np) / np.sum(np.exp(pred_np), axis=1, keepdims=True)
            confidence = float(np.max(softmax_output))
            label_idx = int(np.argmax(softmax_output, axis=1)[0])
            label_str = idx_to_class.get(label_idx, "Unknown")
            
            # 결과 로그 추가
            results_log.append(f"{os.path.basename(image_path)} - Detected Face - Label: {label_str}, Confidence: {confidence:.2f}")

            # 이미지에 박스 및 텍스트 추가
            color = (0, 255, 0)
            cv2.rectangle(img_bgr, (new_x1, new_y1), (new_x2, new_y2), color, 2)
            text_str = f"{label_str}({confidence:.2f})"
            cv2.putText(img_bgr, text_str, (new_x1, new_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    if not detected_any_face:
        print("[Info] 얼굴을 감지하지 못했습니다.")
        
        # 전체 이미지를 RGB로 변환
        crop_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        
        # 전처리 수행
        input_tensor = transform(pil_img).unsqueeze(0)
        input_data = input_tensor.cpu().numpy()
        
        # ONNX 모델 추론
        outputs = onnx_sess.run(None, {input_name: input_data})
        pred_np = outputs[0]
        
        # 소프트맥스 적용
        softmax_output = np.exp(pred_np) / np.sum(np.exp(pred_np), axis=1, keepdims=True)
        confidence = float(np.max(softmax_output))  # 신뢰도 추출
        label_idx = int(np.argmax(softmax_output, axis=1)[0])  # 클래스 인덱스 추출
        label_str = idx_to_class.get(label_idx, "Unknown")  # 클래스 라벨
        
        # 결과 로그 추가
        results_log.append(f"{os.path.basename(image_path)} - Full Image - Label: {label_str}, Confidence: {confidence:.2f}")
        
        # 이미지에 텍스트 추가
        cv2.putText(img_bgr, f"{label_str} (No Face) ({confidence:.2f})", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


    return img_bgr, results_log

# ------------------------------
# 4) 메인 실행
# ------------------------------
if __name__ == "__main__":
    all_results_log = []
    image_windows = []

    # sample 폴더 내 모든 이미지 순회                            
    for filename in os.listdir(image_folder_path):
        file_path = os.path.join(image_folder_path, filename)

        # 이미지 파일만 처리 (jpg, png 등 확장자 확인)
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"\n[INFO] Processing file: {filename}")
            result_img, log = detect_and_classify_faces(
                image_path=file_path,
                detector=face_detector,
                onnx_sess=cls_session,
                extend_ratio=0.4
            )

            if result_img is not None and log is not None:
                all_results_log.extend(log)
                window_name = f"Result - {filename}"
                cv2.imshow(window_name, result_img)
                image_windows.append(window_name)

    # 모든 결과 로그 출력
    print("\n[Final Results Log]")
    for log_entry in all_results_log:
        print(log_entry)

    # 모든 창 닫기 대기
    cv2.waitKey(0)
    cv2.destroyAllWindows()
