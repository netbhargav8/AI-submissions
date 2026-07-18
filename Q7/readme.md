# Reusable Image Preprocessing & Classification Pipeline

An automated computer vision data-engineering pipeline designed to ingest raw image directories, apply uniform spatial adjustments, normalize pixel values, partition data into train/validation arrays, and output a detailed tracking CSV.

## Prerequisites & Structural Installation

This pipeline runs on Python 3. It requires OpenCV (for image processing), Scikit-Learn (for matrix splitting), and Pandas (for CSV logging). Install the workspace dependencies using your terminal:

```bash
pip install opencv-python scikit-learn pandas numpy