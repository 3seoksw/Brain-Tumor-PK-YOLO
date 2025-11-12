# PK-YOLO

> [!NOTE]
> This repository is a **fork** of [PK-YOLO](https://github.com/mkang315/PK-YOLO) [[1]](#1) with some modifications.
> Original credits and license are preserved. We are not affiliated with the authors.
> We appreciate the original authors' work.

#### Installation

```bash
conda create -n <env-name> python==3.12.4
conda activate <env-name>
pip-compile --output-file=requirements.txt requirements.in
pip install -r requirements.txt
```

#### Dataset Preparation

> [!IMPORTANT]
> Before running a program, please go [here](https://github.com/3seoksw/Brain-Tumor-PK-YOLO/tree/main/data/) for preparing the dataset required for the training.

#### Training

```bash
python train_dual.py
```

## Reference

<a id="1" href="">[1]</a>
M. Kang, F. F. Ting, R. C.-W. Phan, and C.-M. Ting, "Pk-yolo: Pretrained knowledge guided yolo for brain tumor detection in multiplane mri slices," in
<i>Proc. Winter Conf. Appl. Comput. Vis. (WACV)</i>, Tucson, AZ, USA, Feb. 28–Mar. 4, 2025, pp. 3732–3741.
