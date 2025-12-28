## YOLO Solution (Ultralytics) — Fruit Detection

All YOLO files are collected here. Run commands from this folder unless stated.

### What it does
- Detects fruits in images/video/webcam with bounding boxes and class labels.

### File map
- Inference script: `ultralytics_fruit/webcam_fruit_det.py`
- Training helper: `fruit_yolo/train_yolo.py`
- Dataset config: `fruit_yolo/data.yaml` (edit paths to your dataset)
- Base weights: `yolov8n.pt` (example starting checkpoint)
- Outputs after training: `runs/detect/<run>/weights/best.pt` (or your chosen run name)
- Extra weights: `weights/best.pt` (existing checkpoint)
- Classification weight: `yolov8n-cls.pt`

### Environment setup
```powershell
cd "D:\Study in Slovakia\3 4 5 Introduction to machine learning\FruitClassification"
.\fcenv\Scripts\Activate.ps1
cd solutions\yolo
pip install -U ultralytics opencv-python
# Optional GPU PyTorch:
# pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Dataset expectation (YOLO format)
- Images: `dataset/images/train`, `dataset/images/val`
- Labels: `dataset/labels/train`, `dataset/labels/val` (same basenames as images; txt with YOLO annotations)
- Update absolute/relative paths in `fruit_yolo/data.yaml` to your dataset before training.
- Classes (in data.yaml): `Apple, Banana, Carambola, Orange, Peach, Pear, Tomatoes`

### Train
```powershell
python fruit_yolo/train_yolo.py --data fruit_yolo/data.yaml ^
  --model yolov8n.pt ^
  --epochs 50 ^
  --batch 16 ^
  --imgsz 640 ^
  --device 0 ^
  --project runs ^
  --name fruit_yolo
```
- If no GPU, set `--device cpu`.
- Result: weights at `runs/detect/fruit_yolo/weights/best.pt`.

### Inference (webcam / images / video / folder / URL)
```powershell
python ultralytics_fruit/webcam_fruit_det.py --model runs/detect/fruit_yolo/weights/best.pt ^
  --source 0 ^
  --conf 0.5 ^
  --imgsz 640 ^
  --device cpu
```
- `--source 0` = default webcam. Use a folder path, video file, or RTSP/URL similarly.
- Use `--device 0` to run on GPU. Press `q` to exit window.

### Notes
- Keep weights (`best.pt`) in `runs` or `weights`; point `--model` to the desired checkpoint.
- If GPU memory is tight, reduce `--batch` or use a smaller base model (e.g., `yolov8n.pt`).
- Ensure every image in the dataset has a matching label file before training.
