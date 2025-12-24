from ultralytics import YOLO
import torch
import os


if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.__version__)

    yaml_path = "dataset/data.yaml" 

    model = YOLO('yolo11s.pt')

    print("Начинаем обучение. Главный процесс запущен.")
    
    results = model.train(
        data=yaml_path,
        epochs=150,
        imgsz=480,
        batch=-1,
        patience=50,
        name='yolo_run',
)
