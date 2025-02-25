# Human Face Detection using YOLOv8

## Overview

This project implements a human face detection system using YOLOv8, a state-of-the-art object detection model. It leverages Python, OpenCV, Pandas, and PyTorch to preprocess a dataset of face images, train a YOLOv8 model, and evaluate its performance. The trained model can then be used for real-time face detection in images and videos.

## Features

-   **Data Preprocessing:** Splits the dataset into training, validation, and test sets for robust model training and evaluation.
-   **Bounding Box Labeling:** Generates YOLO-compatible annotation files from a CSV file containing bounding box coordinates.
-   **YOLOv8 Training:** Utilizes the Ultralytics YOLOv8 framework for efficient and accurate model training.
-   **Model Evaluation:** Calculates the mean Average Precision at a threshold of 0.5 (mAP@50) to assess the model's accuracy on the validation and test datasets.
-   **Visualization:** Provides tools to visualize bounding box predictions on images and plot training performance curves.
-   **Model Export:** Saves the trained model in the `best.pt` format, which can be easily deployed for real-time inference.

## Dataset Structure

The project assumes the following dataset structure:


-   `images/`: Contains the image files, organized into `train`, `validation`, and `test` subdirectories.
-   `labels/`: Contains the corresponding annotation files in YOLO format (`.txt` files), also organized into `train`, `validation`, and `test` subdirectories. Each `.txt` file corresponds to an image and contains the bounding box coordinates and class label for each face in the image.
-   `faces.csv`: A CSV file containing the bounding box annotations in the following format: `image_id, x_min, y_min, x_max, y_max`.  This file is used to generate the YOLO format annotation files.

## Installation and Setup

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/HumanFace_Detection.git
    cd HumanFace_Detection
    ```

2.  **Install Dependencies**

    You can install the required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

    Alternatively, you can install the packages manually:

    ```bash
    pip install numpy pandas opencv-python matplotlib ultralytics wandb
    ```

## Training the YOLOv8 Model

1.  **Install Ultralytics YOLOv8**

    ```bash
    pip install ultralytics
    ```

2.  **Train the Model**

    The following code snippet demonstrates how to train the YOLOv8 model using the Ultralytics library:

    ```python
    from ultralytics import YOLO

    # Load a YOLOv8 model.  You can also specify a path to a custom model, e.g., 'path/to/best.pt'
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # Build a new model from scratch and transfer weights

    # Train the model
    results = model.train(data='config.yaml', epochs=4, resume=True, iou=0.5, conf=0.001)
    ```

    -   `yolov8n.yaml`:  This specifies the model architecture.  `yolov8n` is the Nano version, which is the smallest and fastest.  You can choose other sizes like `yolov8s`, `yolov8m`, `yolov8l`, or `yolov8x` for potentially higher accuracy but slower performance.
    -   `yolov8n.pt`: This loads pre-trained weights from the YOLOv8 Nano model.  This is optional but highly recommended for faster convergence and better results.
    -   `data='config.yaml'`: Specifies the path to the data configuration file (`config.yaml`), which defines the dataset's structure, class names, and other training parameters.  **You will need to create this file.**  See the "Data Configuration File (config.yaml)" section below.
    -   `epochs=4`: Sets the number of training epochs.  Increase this for better results, but it will take longer to train.
    -   `resume=True`:  If training is interrupted, this will resume from the last saved checkpoint.
    -   `iou=0.5`: Sets the Intersection over Union (IoU) threshold for considering a prediction as correct.
    -   `conf=0.001`: Sets the confidence threshold for filtering predictions.

### Data Configuration File (config.yaml)

You need to create a `config.yaml` file that defines the dataset's structure and other training parameters. Here's an example:

```yaml
train: Human_Faces/images/train  # Path to the training images
val: Human_Faces/images/validation # Path to the validation images
test: Human_Faces/images/test # Path to the test images

nc: 1  # Number of classes (1 for face detection)

names: ['face']  # Class names
