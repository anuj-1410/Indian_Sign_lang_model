# Indian Sign Language Detection

This project is a machine learning-based application designed to detect and recognize Indian Sign Language (ISL) gestures, including both alphabets and digits, using a webcam. The application uses computer vision techniques and a deep learning model to classify hand gestures in real-time.

---

## Features

- **Real-Time Gesture Recognition**: Detects and recognizes hand gestures for alphabets (`a-z`) and digits (`0-9`) using a webcam.
- **Interactive UI**: A Streamlit-based user interface for easy interaction.
- **Custom Dataset**: Supports training on a custom dataset of ISL gestures.
- **Preprocessing Pipeline**: Includes preprocessing steps for resizing, cropping, and normalizing images.
- **Model Training**: A convolutional neural network (CNN) is used for gesture classification.
- **Text Output**: Converts recognized gestures into text for display.

---

## Folder Structure

### Dataset

**These Datasets available on my github**

- **`Final_Sign_Dataset/`**: Contains raw images of hand gestures organized into subfolders for each class (e.g., `0`, `a`, `b`, etc.).
- **`preprocessed_alphaDigi_dataset/`**: Contains preprocessed and split datasets for training, validation, and testing.

### Notebooks

- **`dataCollection.ipynb`**: Script for collecting and saving hand gesture images using a webcam.
- **`modelTraining.ipynb`**: Notebook for training the CNN model on the preprocessed dataset.
- **`testingOpenCV.ipynb`**: Script for testing the trained model with real-time webcam input.

### Application

- **`appUI.py`**: Streamlit-based application for real-time gesture recognition and text generation.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anuj-1410/Indian_Sign_lang_model.git
   ```

# Usage

## 1. Data Collection

To collect hand gesture images for training:

Run the dataCollection.ipynb notebook.
Use the webcam to capture images of hand gestures for each class (e.g., a, b, c, etc.).
Press s to save an image and q to quit.

## 2. Model Training

To train the model:

Run the modelTraining.ipynb notebook.
The dataset will be split into training, validation, and test sets.
The trained model will be saved as sign_language_alphaDigi_model.h5.

## 3. Real-Time Testing

To test the model in real-time:

```bash
   streamlit run appUI.py
```

If it doesn't work use this:

```bash
   streamlit run appUI.py --server.port 8888
```

Run the appUI.py script:
Use the Streamlit interface to open the camera and start recognizing gestures.

# Model Architecture

The model is a Convolutional Neural Network (CNN) with the following layers:

Convolutional Layers: Extract spatial features from images.
MaxPooling Layers: Reduce spatial dimensions.
Batch Normalization: Normalize activations for faster convergence.
Dropout: Prevent overfitting.
Dense Layers: Fully connected layers for classification.

## Requirements

Python 3.8+
OpenCV
TensorFlow
Streamlit
cvzone
NumPy
Matplotlib

## Sample images

<div style="display: flex; align-items: centre;">
   <h3>Images of number 0, 1, 2</h3>
   <img src="sample hand images/0.jpg" alt="img 1" style="width: 30%; height: auto; margin-right: 10px">
   <img src="sample hand images/1.jpg" alt="img 2" style="width: 30%; height: auto; margin-right: 10px">
   <img src="sample hand images/2.jpg" alt="img 3" style="width: 30%; height: auto;">
</div>

<div style="display: flex; align-items: centre;">
   <h3>Images of number a, b, c</h3>
   <img src="sample hand images/a.jpg" alt="img 1" style="width: 30%; height: auto; margin-right: 10px">
   <img src="sample hand images/b.jpg" alt="img 2" style="width: 30%; height: auto; margin-right: 10px">
   <img src="sample hand images/c.jpg" alt="img 3" style="width: 30%; height: auto;">
</div>
