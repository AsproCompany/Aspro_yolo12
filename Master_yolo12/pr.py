from ultralytics import YOLO

model = YOLO("Aspro_yolo12.pt")

model.predict(source="afriq0.MP4", show=True, save=True)