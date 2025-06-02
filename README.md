# Manual Edge Detection in Python

This project implements a complete edge detection pipeline from scratch using Python and OpenCV.
It includes Gaussian blur, Sobel filtering, gradient direction quantization, non-maximum suppression,
double thresholding, and hysteresis—similar to the Canny edge detector.

## Features

- Gaussian Blur (Manual kernel generation)
- Sobel Operator (Gx, Gy, Gradient magnitude)
- Gradient Direction in Degrees
- Direction Quantization (0°, 45°, 90°, 135°)
- Non-Maximum Suppression
- Double Thresholding
- Hysteresis
- Visualization using matplotlib

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

## Usage

Replace `test.png` with your input image.

```bash
python main.py
```

## Author

Alireza Jafari  
GitHub: [@jafarirezaali](https://github.com/jafarirezaali)