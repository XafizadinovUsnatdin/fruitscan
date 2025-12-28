## CNN Solution (TensorFlow/Keras) — Fruit Classification

All CNN files are collected here. Run commands from this folder unless stated.

### What it does
- Classifies a single cropped fruit image into one class.
- Uses saved weights `fruit_cnn.h5` and class map `class_indices.json`.

### File map
- Code: `fruitclassification/run_fruit_cam.py` (train/predict/webcam), `fruitclassification/predict_from_folder.py`, `fruitclassification/webcam_fruit_cnn.py` (lite webcam).
- Weights: `fruit_cnn.h5`.
- Class map: `class_indices.json`.
- Dataset: `train/train/<class_name>/*.jpg`, `test/test/*.jpg`.

### Environment setup
```powershell
cd "D:\Study in Slovakia\3 4 5 Introduction to machine learning\FruitClassification"
.\fcenv\Scripts\Activate.ps1
cd solutions\cnn
pip install -U tensorflow==2.16.1 opencv-python numpy pillow matplotlib
```
If TensorFlow has issues on Python 3.13, create a new venv with Python 3.11/3.10 and reinstall.

### Dataset expectation
ImageNet-style folders:
- Training root (default): `train/train/<class_name>/*.jpg`
- Validation split is created automatically inside `run_fruit_cam.py` via `--val-split` (default 0.15).
- Test images: example in `test/test/`.

### Train
```powershell
python fruitclassification/run_fruit_cam.py --mode train ^
  --data-root train/train ^
  --epochs 20 ^
  --img-size 100 ^
  --batch-size 32 ^
  --model-path fruit_cnn.h5 ^
  --classes-path class_indices.json ^
  --target-acc 0.9
```
Outputs: `fruit_cnn.h5` (best weights) and `class_indices.json` (class-to-index map).

### Inference (single image)
```powershell
python fruitclassification/run_fruit_cam.py --mode predict ^
  --image test/test/0001.jpg ^
  --model-path fruit_cnn.h5 ^
  --classes-path class_indices.json ^
  --img-size 100
```

### Webcam demo
```powershell
python fruitclassification/run_fruit_cam.py --mode webcam ^
  --model-path fruit_cnn.h5 ^
  --classes-path class_indices.json ^
  --img-size 100 ^
  --camera-index 0
```
Press `q` to quit.

### Batch folder prediction
```powershell
python fruitclassification/predict_from_folder.py --images test/test ^
  --model-path fruit_cnn.h5 ^
  --classes-path class_indices.json ^
  --img-size 100 ^
  --limit 0
```

### Notes
- Keep `fruit_cnn.h5` and `class_indices.json` in this folder; scripts expect these defaults when run here.
- Keep train and predict `--img-size` consistent if you customize it.
