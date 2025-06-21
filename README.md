<h1 align="center"><strong>HugWBC: A Unified and General Humanoid Whole-Body Controller for Versatile Locomotion</strong></h1>
<p align="center">
  <a href=''>Yufei Xue*</a> &nbsp;&nbsp;
  <a href='https://apex.sjtu.edu.cn/members/wentaodong@apexlab.org'>Wentao Dong*</a> &nbsp;&nbsp;
  <a href='https://minghuanliu.com'>Minghuan Liu^</a> &nbsp;&nbsp;
  <a href='https://wnzhang.net/'>Weinan Zhang</a> &nbsp;&nbsp;
  <a href='https://oceanpang.github.io/'>Jiangmiao Pang</a> &nbsp;&nbsp;
</p>
<p align="center">
* Equal contribution&nbsp;&nbsp;&nbsp;&nbsp;^ Project Lead
</p>

<p align="center">
    <img src="./imgs/sjtu.png" height=100"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 
    <img src="./imgs/share-logo.png" height="100">
</p>

<div align="center">

<h3 align="center">  Robotics: Science and Systems (RSS) 2025 </h3>

<p align="center">
<h3 align="center">
<a href="https://hugwbc.github.io/">Website</a> | 
<a href="https://arxiv.org/abs/2502.03206/">arXiv</a> | 
<a href="https://www.youtube.com/watch?v=JP9A0EIu7nc">Video</a> 
  <div align="center"></div>
</p>

<p align="center">
    <img src="./imgs/framework.png" width=90%" style="margin-right: 100px;"></img>  
</p>

</div>

## 🔥 News

- \[2025-06\] We opensource traning code of HugWBC
- \[2025-02\] We release the [paper](https://arxiv.org/abs/2502.03206) and demos of HugWBC.

## 📚 Installation
Create mamba/conda environment.

```bash
conda create -n hugwbc python=3.8
conda activate hugwbc
pip3 install torch torchvision torchaudio
```

Download [IsaacGym](https://developer.nvidia.com/isaac-gym/download) and extract:

```bash
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4
cd isaacgym/python
pip isntall -e .
```

Then you should first clone this repository to your Ubuntu computer by running:

```bash
git clone https://github.com/OpenRobotLab/HugWBC.git 
cd HugWBC/rsl_rl
pip install -e . 
```

Train your robot as follows:

```bash
cd HugWBC
python legged_gym/scripts/train.py --task=h1int --headless 
```
Visualization of training results

```bash
python legged_gym/scripts/play.py --task=h1int
```

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{xue2025unified,
  title={A Unified and General Humanoid Whole-Body Controller for Fine-Grained Locomotion}, 
  author={Xue, Yufei and Dong, Wentao and Liu, Minghuan and Zhang, Weinan and Pang, Jiangmiao},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2025},
  }
```

</details>

## 👏 Acknowledgements

- [RSL_RL](https://github.com/leggedrobotics/rsl_rl).
- [Legged_gym](https://github.com/leggedrobotics/rsl_rl).
- [Walk-These-Ways](https://github.com/leggedrobotics/rsl_rl).
