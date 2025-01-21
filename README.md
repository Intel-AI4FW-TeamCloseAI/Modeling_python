# Modeling_python
Includes dataset image processing and model fine-tuning

## Image dataset used
- Detect AI-Generated Faces: https://www.kaggle.com/datasets/shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset
- deepfake and real images: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
- FaceForensics: https://www.kaggle.com/datasets/greatgamedota/faceforensics

## Model used
- efficientnet_b0: https://pytorch.org/vision/master/models/generated/torchvision.models.efficientnet_b0.html
- yolov11n-face: https://github.com/akanametov/yolo-face

## Classification (Binary, 0: Fake, 1: Real)
Dataset
- Train
  - Fake
  - Real
- Validation
  - Fake
  - Real
- Test
  - Fake
  - Real
  
## Image preprocessing
- Resize (128, 128)
- Data Augmentation
- Normalize: mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]

## Fine-tuning
- Criterion: CrossEntropyLoss()
- Optimizer: Adam (lr = 1e-4)
- Epochs: 50 w/ Early Stopping (validation loss, patience = 5)
