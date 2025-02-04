                                        ////YOLOv8 Object Detection Training in Google Colab////

This project trains a YOLOv8 object detection model using a custom dataset in Google Colab. 
The model is optimized for detecting five classes and utilizes the YOLOv8s pre trained model for efficient and accurate detection.

                                                         Setup Instructions

1. Install Dependencies

Ensure you have the required libraries installed:

!pip install ultralytics
!pip install -U albumentations

2. Verify YOLOv8 Installation

Confirm successful installation:

from ultralytics import YOLO
YOLO('yolov8s.pt')

3. Load and Configure Dataset

Place your dataset in the correct directory.

Ensure the dataset YAML file (data.yaml) is correctly structured with class names and image paths.

Example data.yaml:

train: path/to/train/images
val: path/to/val/images
test: path/to/test/images
nc: 5
names: ['class1', 'class2', 'class3', 'class4', 'class5']

4. Train the Model

Run the following command to start training:

!yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=25 imgsz=224 plots=True

task: Detection task

model: YOLOv8s pre-trained weights

data: Dataset YAML file

epochs: Number of training epochs (adjust as needed)

imgsz: Image size (default: 224)

plots: Enables training plots

5. Monitor Training Progress

To visualize training logs in real-time, use TensorBoard:

%load_ext tensorboard
%tensorboard --logdir runs/detect/train

6. Validate the Model

After training, evaluate model performance:

!yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml

7. Inference with the Trained Model

                                                           Test the model on new images:

!yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source='test_image.jpg' save=True

                                                          //Troubleshooting & Recommendations//

Dataset Issues: Ensure images and labels are correctly formatted in YOLO format.

Memory Errors: Reduce batch size or image size if running out of memory.

Performance Optimization: Try adjusting hyperparameters such as learning rate, optimizer, and augmentation techniques.

                                                                    //References//

Ultralytics YOLOv8 Documentation

Google Colab Guide

