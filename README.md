# Double Check My Desired Return

This is the official code base for the paper `Double Check My Desired Return: Transformer with Target Alignment for Offline Reinforcement Learning`. 


## Contents

- [Instructions](#Instructions)
- [Quick Start](#quick-start)
- [Citation](#citation)
- [Acknowledgements](#acknowledgments)


## Instructions

### Install python packages from scratch
Make a new conda env
```
conda create -n doctor python=3.9
conda activate doctor
```

Run these commands to install all dependencies
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```


## Quick Start
When your environment is ready, you could run scripts in the "scripts/doctor.py". For example:
``` 
python run_doctor.py --path /your_path/Doctor/research/config/doctor.py
```

## Citation
If you find our work useful, please feel free to cite our work.


## Acknowledgments

This work builds on top of the following dependencies.
 * [facebookresearch/mtm](https://github.com/facebookresearch/mtm): Masked Trajectory Model, which this work builds upon.
 * [charleshsc/QT](https://github.com/charleshsc/QT): Q-value Regularized Transformer for Offline Reinforcement Learning.
 * [tinkoff-ai/CORL](https://github.com/tinkoff-ai/CORL): CORL provides an Offline Reinforcement Learning library.

Thanks for their great works.