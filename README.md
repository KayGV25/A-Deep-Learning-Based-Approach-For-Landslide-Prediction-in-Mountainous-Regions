# A Deep Learning-Based Approach for Landslide Prediction in Mountainous Regions

This project trains PyTorch image classification models for landslide prediction. The training script expects a preprocessed NumPy memmap dataset created from the notebooks.

Dataset access: [Final dataset](https://mailbcuac-my.sharepoint.com/:u:/g/personal/dang_vu2_mail_bcu_ac_uk/IQDQo1nJit0iTJLyS1kDQjnTAWymfqJof6lh1Jm9g8OVjKg?e=RDCtx6)

For running the preprocssing please install these followig dataset:
- [LMHLD](https://zenodo.org/records/17258777/files/LMHLD.rar?download=1)
- [Bijie](https://gpcv.whu.edu.cn/data/Bijie_landslide_dataset.zip)
- [CAS](https://zenodo.org/api/records/10294997/files-archive) (Remember to extract all files)

## Tested Training Specification

| Component | Specification |
| --- | --- |
| GPU | NVIDIA ASUS L40, 20 GB VRAM |
| CPU | 8-core processor |
| RAM | 36 GB system memory |
| Framework | PyTorch |
| Language | Python |
| Acceleration | CUDA with mixed precision training |
| OS | Linux server environment |

## Project Files

| File | Purpose |
| --- | --- |
| `data_preprocessing.ipynb` | Builds the 3-class dataset and exports memmap files for training. |
| `Bijie dat binary.ipynb` | Builds the Bijie binary dataset and exports memmap files. |
| `train.py` | Trains the configured models from a memmap dataset. |
| `requirements.txt` | Python package list. |

## 1. Create and Activate a Virtual Environment

Run these commands from the project root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2. Install Dependencies

Install the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

If PyTorch CUDA wheels are not found by your default package index, install the CUDA build that matches your server from the official PyTorch wheel index, then rerun the requirements install. For the current requirement pins, the project expects CUDA-enabled `torch` and `torchvision`.

The current training script also imports packages that are not listed in `requirements.txt`. Install them before training:

```bash
pip install scikit-learn kornia
```

For running all notebook cells, these packages may also be needed:

```bash
pip install jupyter matplotlib umap-learn
```

## 3. Required Folders

Create a dataset workspace and a model output folder. The code currently uses Linux absolute paths in `train.py`. For example:

```bash
mkdir -p /home/islab/data/khuong/final_dataset_dat
mkdir -p /home/islab/data/khuong/models
```

If you use different folders, edit these constants near the top of `train.py`:

```python
DATASET_DIR = "/home/islab/data/khuong/final_dataset_dat"
MODEL_DIR = "/home/islab/data/khuong/models"
```

The training dataset folder must contain these files:

```text
final_dataset_dat/
  X_train.dat
  y_train.dat
  X_test.dat
  y_test.dat
  meta_train.npy
  meta_test.npy
```

`train.py` saves checkpoints into `MODEL_DIR` as:

```text
resnet18_best.pth
mobilenet_best.pth
efficientnet_b3_best.pth
stack_ensembled_best.pth
```

## 4. Prepare the Dataset

### 3-class dataset

Open and run:

```bash
jupyter notebook data_preprocessing.ipynb
```

Important notebook paths to check before running:

```python
LMHLD_ROOT = "./dataset/LMHLD/Comparison_dataset_same_patch_size"
OUTPUT = "./dataset/final_dataset_dat"
TRAIN_DIR = "./dataset/final_dataset_split/train"
TEST_DIR = "./dataset/final_dataset_split/test"
```

The final output folder must contain:

```text
X_train.dat
y_train.dat
X_test.dat
y_test.dat
meta_train.npy
meta_test.npy
```

After preprocessing, either copy that output folder to the `DATASET_DIR` used in `train.py`, or change `DATASET_DIR` to point to the generated folder.

### Bijie binary dataset

Open and run:

```bash
jupyter notebook "Bijie dat binary.ipynb"
```

Important notebook paths:

```python
SOURCE_DIR = "./dataset/Bijie-landslide-dataset"
DEST_DIR = "./dataset/bijie_binary_split"
OUTPUT_DIR = "./dataset/bijie_binary_dat"
```

If training on the binary dataset, update `train.py` before running:

```python
NUM_CLASSES = 2
DATASET_DIR = "/path/to/bijie_binary_dat"
MODEL_DIR = "/path/to/models/bijie_binary"
```

## 5. Configure Training

`train.py` does not currently use command-line arguments. Edit the constants near the top of the file:

```python
IMG_SIZE = 224
BATCH_SIZE = 96
NUM_WORKERS = 6
NUM_CLASSES = 3
EPOCHS = 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "/home/islab/data/khuong/final_dataset_dat"
MODEL_DIR = "/home/islab/data/khuong/models"
```

The training setup used for the reported run was:

```text
BATCH_SIZE = 96
NUM_WORKERS = 6
NUM_CLASSES = 3
EPOCHS = 60
Mixed precision = enabled
CUDA = enabled
```

Choose which models to train by editing the `model_names` list near the bottom of `train.py`:

```python
model_names = [
    "resnet18",
    "mobilenet",
    "efficientnet_b3",
    "stack_ensembled",
]
```

Note: `stack_ensembled` loads the saved checkpoints for `resnet18`, `mobilenet`, and `efficientnet_b3`. Train those three models first before training `stack_ensembled`.

## 6. Run Training

From the project root:

```bash
source .venv/bin/activate
python train.py
```

The script will:

1. Load `meta_train.npy` and `meta_test.npy`.
2. Open `X_train.dat`, `y_train.dat`, `X_test.dat`, and `y_test.dat` with `np.memmap`.
3. Split the training set into train and validation subsets.
4. Train each model listed in `model_names`.
5. Save the best checkpoint for each model into `MODEL_DIR`.

## 7. Common Changes

To reduce GPU memory usage:

```python
BATCH_SIZE = 32
```

To reduce CPU dataloader load:

```python
NUM_WORKERS = 2
```

To train only one model:

```python
model_names = ["efficientnet_b3"]
```

To train the ensemble after base models are saved:

```python
model_names = ["stack_ensembled"]
```

## 8. Expected Dataset Class Maps

For the 3-class dataset:

```python
CLASS_MAP = {
    "non_landslide": 0,
    "low_risk": 1,
    "high_risk": 2,
}
```

Make sure `NUM_CLASSES` in `train.py` matches the dataset class map.
