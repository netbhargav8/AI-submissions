# Face Mask Detection using MobileNetV2

A deep learning-based Face Mask Detection system built using **TensorFlow** and **MobileNetV2** with transfer learning. The model classifies images into **With Mask** and **Without Mask** categories using a pretrained convolutional neural network.

## Features

- Transfer Learning using MobileNetV2
- Image preprocessing and data augmentation
- Binary classification (With Mask / Without Mask)
- Model evaluation using Accuracy, Precision, and Recall
- Inference speed measurement
- Model saved in Keras (`.keras`) format
- Ready for edge deployment optimization

## Dataset

The project uses a face mask dataset with the following structure:

```
FaceMaskDataset/
│
├── with_mask/
└── without_mask/
```

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Project Workflow

```
Dataset
   │
   ▼
Preprocessing & Augmentation
   │
   ▼
MobileNetV2 (Pretrained)
   │
   ▼
Transfer Learning
   │
   ▼
Training
   │
   ▼
Evaluation
   │
   ▼
Inference
```

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/FaceMaskDetection.git
cd FaceMaskDetection
```

Install the required dependencies:

```bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn
```

## Training

Place the dataset inside the project directory and run:

```bash
python train.py
```

The trained model will be saved as:

```
FaceMask_MobileNetV2.keras
```

## Testing

Run the testing script:

```bash
python test.py
```

Example output:

```
Prediction : With Mask
Confidence : 98.72%
```

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- Classification Report
- Inference Speed (FPS)

## Edge Deployment Optimizations

- TensorFlow Lite (TFLite) conversion
- INT8 Quantization
- Model Pruning
- Reduced input resolution
- Deployment on Raspberry Pi, Jetson Nano, or Android devices

## Project Structure

```
FaceMaskDetection/
│
├── FaceMaskDataset/
├── train.py
├── test.py
├── FaceMask_MobileNetV2.keras
├── README.md
└── requirements.txt
```

## Future Improvements

- Fine-tune deeper MobileNetV2 layers
- Real-time webcam detection
- Incorrect mask detection
- TensorFlow Lite deployment

## License

This project is intended for educational and academic purposes.