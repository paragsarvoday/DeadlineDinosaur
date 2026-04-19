# DeadlineDinosaur 🦕

> **🥈 2nd Place — 3D Gaussian Splatting Challenge, SIGGRAPH Asia 2025**
> *Average PSNR: 28.04 dB · Reconstruction Time: 57s (within the 60s constraint) · Single RTX 4090 GPU*

[Parag Sarvoday Sahu](https://paragsarvoday.github.io/)<sup>1</sup>,
[Vishwesh Vhavle](https://vishweshvhavle.github.io/)<sup>2</sup>,
[Kshitij Aphale](https://github.com/lighterbird)<sup>2</sup>,
and
[Avinash Sharma](https://3dcomputervision.github.io/about/)<sup>2</sup>

<sup>1</sup>IIT Gandhinagar &nbsp;|&nbsp; <sup>2</sup>IIT Jodhpur

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Challenge](https://img.shields.io/badge/SIGGRAPH%20Asia%202025-3DGS%20Challenge%20%F0%9F%A5%88%202nd%20Place-silver)](https://gaplab.cuhk.edu.cn/projects/gsRaceSIGA2025/index.html#evaluation)

---

## About

DeadlineDinosaur is our submission to the [3D Gaussian Splatting Challenge](https://gaplab.cuhk.edu.cn/projects/gsRaceSIGA2025/) held at the **3DGS Workshop, SIGGRAPH Asia 2025** (Hong Kong, December 15–18, 2025), organized by the GAP Lab at CUHK.

The challenge tasked participants with achieving high-quality novel-view synthesis via 3DGS under extremely tight latency, full reconstruction of a scene within **60 seconds on a single GPU**. We ranked **2nd** out of 5 finalists with an average PSNR of **28.04 dB** and a reconstruction time of **57 seconds**.

| Rank | Team | Affiliation | Avg. PSNR (dB) |
|:----:|------|-------------|:--------------:|
| 🥇 1st | 3DV-CASIA | Institute of Automation, CAS & Wuhan University | — |
| 🥈 **2nd** | **DeadlineDinosaur** | **IIT Gandhinagar & IIT Jodhpur** | **28.04** |
| 🥈 2nd | MT-AI | Moore Threads | — |
| 🥉 3rd | AwesomeGS | Hangzhou Dianzi University | — |
| 🥉 3rd | XT | Tianjin University | — |

Our technical report is available in `DeadlineDinosaur_3DGS2025_Paper.pdf`.

---

## 1. Installation

### Make sure the current working directory is the project folder
```bash
cd DeadlineDinosaur
```

### Create conda environment using `environment.yml`
> Note: This can take a while.
```bash
conda env create --file environment.yml
conda activate DeadlineDinosaur
```

### Install all submodules
```bash
pip install deadlinedino/submodules/simple-knn
pip install deadlinedino/submodules/fused_ssim
pip install deadlinedino/submodules/gaussian_raster
```

### Repository Structure
```
DeadlineDinosaur/
├── deadlinedino
│   ├── submodules
│   │   ├── fused_ssim
│   │   ├── gaussian_raster
│   │   └── simple-knn
├── data
├── outputs
├── environment.yml
├── train.py
├── evaluate.py
├── README.md
├── LICENSE.md
├── LiteGS_LICENSE.md
└── DashGaussian_LICENSE.md          
```

---

## 2. Data Preparation
```bash
mkdir data
```
Place the `eval_data_pinhole/` directory inside `data/`.

The data folder should look like:

```
data/
└── eval_data_pinhole             # Processed dataset folders (13 total)   
    ├── 1747834320424            
    │   ├── images_gt_downsampled # Extracted frames (.jpg)
    │   │   ├── 000000.png
    │   │   ├── ...\
    │   │   └── 000199.png
    ├── sparse                    # Sparse reconstruction data
    │   │   └── 0
    │   │       ├── cameras.txt
    │   │       ├── frames.txt
    │   │       ├── images.txt
    │   │       ├── points3D.ply
    │   │       ├── points3D.txt
    │   │       ├── project.ini
    │   │       └── rigs.txt
    │   └── train_test_split.json
    ├── 1748153841908
    ├── 1748165890960
    ├── 1748242779841
    ├── 1748243104741
    ├── 1749449291156
    ├── 1749606908096
    ├── 1749803955124
    ├── 1750578027423
    ├── 1750824904001
    ├── 1750825558261
    ├── 1750846199351
    ├── 1751090600427
    └── ReadMe_Round2.md
```

---

## 3. Training

Run with default structure:
```bash
python train.py
```

Run with custom paths and GPU:
```bash
python train.py \
    --dataset_dir data/eval_data_pinhole \
    --output_dir outputs \
    --gpu 0
```

---

## 4. Evaluation

Run with default structure:
```bash
python evaluate.py
```

Run with custom paths and GPU:
```bash
python evaluate.py \
    --dataset_dir data/eval_data_pinhole \
    --output_dir outputs \
    --output_run_dir outputs \       # Optional: evaluates the most recent outputs directory by default.
    --gpu 0
```

> **Note:** Evaluation takes a while. Scene-wise rendered and GT images will be saved in the outputs directory. Scene-wise and average PSNRs are printed after the full evaluation completes. Scene-wise training times are also saved as `.json` files alongside the `.ply` files.

---

## Citation

If you find our code or paper useful, please consider citing:
```bibtex
@misc{sahu2025deadlinedinosaur,
  title     = {DeadlineDinosaur: Fast Gaussian Splatting for SIGGRAPH Asia's 3D Gaussian Splatting Challenge},
  author    = {Parag Sarvoday Sahu and Vishwesh Vhavle and Kshitij Aphale and Avinash Sharma},
  year      = {2025},
  url       = {https://github.com/paragsarvoday/DeadlineDinosaur},
}
```

---

## Contact

Contact [Parag Sarvoday Sahu](mailto:parag.sahu@iitgn.ac.in) for questions, comments, and bug reports, or open a [GitHub Issue](https://github.com/paragsarvoday/DeadlineDinosaur/issues).

---

## License

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

---

## 🙏 Acknowledgements

This work builds upon:

- [DashGaussian](https://github.com/YouyuChen0207/DashGaussian) (CVPR 2025)
- [LiteGS](https://github.com/MooreThreads/LiteGS)
