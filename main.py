from ultralytics import YOLO
import os

# Load a model
model = YOLO("C:/Нейронки/best.onnx")

# Train the model
train_results = model.train(
    data="data.yaml", 
    epochs=7,  
    imgsz=640,  
    device="gpu",  
)

metrics = model.val()

folder_path = ("../images/train")
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(folder_path, filename)

        results = model(img_path)

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model