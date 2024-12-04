# If3D: Enhancing Facial Expression Inference with 3D Face Representations via Intermediate Fusion

This repository contains the implementation of intermediate fusion strategies to integrate 3D facial representations into facial expression inference (FEI). The project evaluates the performance of different fusion architectures on the AffectNet dataset and leverages the SMIRK 3D face regression model.

## Repository Structure

`Dataset/` contains the AffectNet dataset files required for training and testing. `Dataset.zip` is the zipped version of the dataset folder for portability. `FinalCheckPoints/` stores the final model checkpoints for each fusion strategy. `FusionConfusionMatrix/` contains confusion matrices generated during the evaluation of different fusion methods. `networks/` includes the model architectures for the backbone and fusion strategies. `picture_*_checkpoints/` stores intermediate checkpoints for different fusion strategies: `team_34_picture_attention_fusion_va_*` for attention-based fusion, `team_34_picture_linear_fusion_va_*` for linear fusion, `team_34_picture_attention_fusion_va_gate_*` for gate-based fusion, and `team_34_picture_attention_fusion_va_SPP_*` for Spatial Pyramid Pooling (SPP) fusion checkpoints. `pretrained/` contains pretrained models and weights used for initializing the networks. `PureSmirkcheckpoints/` includes checkpoints for the SMIRK-only inference without fusion. Python scripts include `affectnet_*_train.py` for training specific fusion strategies, `team_34_affectnet_*_test.py` for testing specific fusion strategies, and `Smirk_batch_Render.py` for generating SMIRK-based 3D representations.

## Installation

Install the required dependencies with `pip install -r requirements.txt`.

## Database and Checkpoints  
Due to the large file size, please download the files from my Google Drive: [Google Drive Link](https://drive.google.com/drive/folders/1rvtPt4OlVsmJnp8LObwKhHBse0PqA9r6?usp=drive_link).  
You will find two files: `Dataset` and `FinalCheckPoints`. Extract these files and replace the `Dataset` and `FinalCheckPoints` folders in this repository to replicate my training and testing experiments.  

## FusionConfusionMatrix  
All ConfusionMatrix plots are located in the `FusionConfusionMatrix` folder. You can view my results directly from there.  

## Usage

To train a model with a specific fusion strategy, run the corresponding training script. For example, to train with attention-based fusion, use `python team_34_affectnet_Origin_Smirk_va_attention_alignment_train.py`. To evaluate a trained model, run the corresponding testing script, such as `python team_34_affectnet_Origin_Smirk_va_SPP_alignment_test.py` for SPP-based fusion.

## Fusion Strategies

This repository supports the following fusion strategies: Linear Fusion, Attention-Based Fusion, Gate-Based Fusion, and Spatial Pyramid Pooling (SPP) Fusion. Each strategy is implemented and tested in separate scripts.

## Checkpoints

Intermediate and final checkpoints are stored in their respective directories. These checkpoints can be used to resume training or for testing.

## Results

Performance metrics for different fusion strategies, including Accuracy, F1 Score, Precision, and Recall, are logged during training and testing. Confusion matrices are stored in `FusionConfusionMatrix/`.

## Reference

**Baseline Model**:  
A Dual-Direction Attention Mixed Feature Network for Facial Expression Recognition  
Official Implementation: [GitHub Repository](https://github.com/SainingZhang/DDAMFN?tab=readme-ov-file)  

Citation:  
@article{zhang2023dual,  
  title={A dual-direction attention mixed feature network for facial expression recognition},  
  author={Zhang, Saining and Zhang, Yuhang and Zhang, Ye and Wang, Yufei and Song, Zhigang},  
  journal={Electronics},  
  volume={12},  
  number={17},  
  pages={3595},  
  year={2023},  
  publisher={MDPI}  
}

---

**Smirk Project**:  
GitHub: [Smirk Repository](https://github.com/georgeretsi/smirk)  

Citation:  
@inproceedings{SMIRK:CVPR:2024,  
  title = {3D Facial Expressions through Analysis-by-Neural-Synthesis},  
  author = {Retsinas, George and Filntisis, Panagiotis P., and Danecek, Radek and Abrevaya, Victoria F. and Roussos, Anastasios and Bolkart, Timo and Maragos, Petros},  
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},  
  year = {2024}  
}
