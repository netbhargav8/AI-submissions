# Manufacturing Defect Detection Pipeline

An end-to-end Machine Learning pipeline designed for automated quality control on a factory floor. The system ingests raw camera feeds of metal casting components, applies image augmentation to synthesize varying orientations, flattens the grayscale tensors, and conducts a model duel to identify the optimal classifier for high-speed edge deployment.

## Prerequisites & Installation

This pipeline uses lightweight algorithms suitable for industrial edge hardware. It requires Python 3, OpenCV, Scikit-Learn, and NumPy. 

```bash
pip install opencv-python scikit-learn numpy