# MonoKey: Occlusion-Robust Keypoint-based Monocular 3D Object Detection with Prior Guidance


This repository hosts the official implementation of [MonoKey: Occlusion-Robust Keypoint-based Monocular 3D Object Detection with Prior Guidance] based on the excellent work [MonoDGP](https://github.com/PuFanqi23/MonoDGP). In this work, we propose a novel keypoint-based monocular method called MonoKey.


The official results in the paper:

<table>
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Val, AP<sub>3D|R40</sub></td>   
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod.</td> 
        <td div align="center">Hard</td> 
    </tr>
    <tr>
        <td rowspan="4",div align="center">MonoKey</td>
        <td div align="center">30.61%</td> 
        <td div align="center">23.39%</td> 
        <td div align="center">20.19%</td> 
    </tr>  
</table>



## Installation
1. Clone this project and create a conda environment:
    ```bash
    cd MonoKey

    conda create -n monokey python=3.8
    conda activate monokey
    ```
    
2. Install pytorch and torchvision matching your CUDA version:
    ```bash
    # For example, We adopt torch 1.9.0+cu111
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    ```
    
3. Install requirements and compile the deformable attention:
    ```bash
    pip install -r requirements.txt

    cd lib/models/monokey/ops/
    bash make.sh
    
    cd ../../../..
    ```
 
4. Download [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) datasets and prepare the directory structure as:
    ```bash
    │Monokey/
    ├──...
    │data/kitti/
    ├──ImageSets/
    ├──training/
    │   ├──image_2
    │   ├──label_2
    │   ├──calib
    ├──testing/
    │   ├──image_2
    │   ├──calib
    ```
    You can also change the data path at "dataset/root_dir" in `configs/monokey.yaml`.
    
## Get Started

### Train
You can modify the settings of models and training in `configs/monokey.yaml` and indicate the GPU in `train.sh`:
  ```bash
  bash train.sh configs/monokey.yaml > logs/monokey.log
  ```
### Test
The best checkpoint will be evaluated as default. You can change it at "tester/checkpoint" in `configs/monokey.yaml`:
  ```bash
  bash test.sh configs/monokey.yaml
  ```

## Acknowlegment
This repo benefits from the excellent work [MonoDGP](https://github.com/PuFanqi23/MonoDGP).
