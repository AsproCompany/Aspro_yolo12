from time import time
from ultralytics import YOLO
import torch

model_fast = YOLO("Aspro_yolo12.yaml")
model_std  = YOLO("yolo12.yaml")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_fast.to(device)
model_std.to(device)
img = torch.randn(1, 3, 640, 640).to(device)

for _ in range(10):
    _ = model_std(img, verbose=False)
    _ = model_fast(img, verbose=False)

if device == 'cuda': torch.cuda.synchronize()
t1 = time()
for _ in range(100): 
    model_std(img, verbose=False)
if device == 'cuda': torch.cuda.synchronize()
avg_std = (time() - t1) / 100

if device == 'cuda': torch.cuda.synchronize()
t2 = time()
for _ in range(100):
    model_fast(img, verbose=False)
if device == 'cuda': torch.cuda.synchronize()
avg_fast = (time() - t2) / 100

print(f"--- Results on {device.upper()} ---")
print(f"Original YOLO12: {avg_std:.4f} seconds per image")
print(f"Aspro-YOLO12:    {avg_fast:.4f} seconds per image")
print(f"Speed Improvement: {((avg_std - avg_fast) / avg_std) * 100:.2f}%")