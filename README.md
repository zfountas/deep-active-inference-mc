# Deep active inference agents using Monte-Carlo methods

This source code release accompanies the manuscript:

Z. Fountas, N. Sajid, P. A.M. Mediano and K. Friston "[Deep active inference agents using Monte-Carlo methods](#)", arXiv.

which is currently available in an online pre-print version. If you use this model or the dynamic dSprites environment in your work, please cite our pre-print.

*NOTE: This repository is currently being updated. For a more complete version of the source code please visit again this page after 13/6/2020.*

---

### Demo behavior

<table style="width:100%;">
  <tr>
    <td align="center"><img src="dsprites.gif" width="200" height="200"/></td>
    <td align="center"><img src="animalai.gif" width="200" height="200"/></td>
  </tr>
  <tr>
    <td align="center">Agent trained in the Dynamic dSprites environment</td>
    <td align="center">Agent trained in the Animal-AI environment</td>
  </tr>
</table>

### Requirements
* Programming language: Python 3
* Libraries: tensorflow >= 2.0.0, numpy, matplotlib, scipy, opencv-python
* [dSprites dataset](https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz).

### Instructions

##### Installation

* Initially, make sure the required libraries are installed in your computer. Open a terminal and type
```bash
pip install -r requirements.txt
```

* Then, clone this repository, navigate to the project directory and download the dSrpites dataset by typing
```bash
wget https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
```
or by manually visiting the above URL.

##### Training
* To train an active inference agent to solve the dynamic dSprites task, type
```bash
python train.py
```
This script will automatically generate checkpoints with the optimized parameters of the agent and store this checkpoints to a different sub-folder every 25 training iterations. The default folder that will contain all sub-folders is ```figs_final_model_0.01_30_1.0_50_10_5```. The script will also generate a number of performance figures, also stored in the same folder. You can stop the process at any point by pressing ```Ctr+c```.

##### Testing
* Finally, once training has been completed, the performance of the newly-trained agent can be demonstrated in real-time by typing
```bash
python test_demo.py -n figs_final_model_0.01_30_1.0_50_10_5/checkpoints/ -m
```
This command will open a graphical interface which can be controlled by a number of keyboard shortcuts. In particular, press:

  * `q` or `esc` to exit the simulation at any point.
  * `1` to enable the MCTS-based full-scale active inference agent (enable by default).
  * `2` to enable the active inference agent that minimizes expected free energy calculated only for a single time-step into the future.
  * `3` to make the agent being controlled entirely by the habitual network (see manuscript for explanation)
  * `4` to activate *manual mode* where the agents are disabled and the environment can be manipulated by the user. Use the keys `w`, `s`, `a` or `d` to move the current object up, down, left or right respectively.
  * `5` to enable an agent that minimizes the terms `a` and `b` of equation 8 in the manuscript.
  * `6` to enable an agent that minimizes only the term `a` of the same equation (reward-seeking agent).
  * `m` to toggle the use of sampling in calculating future transitions.


  ### Bibtex
  ```
  @article{fountas2020daimc,
    title={Deep active inference agents using Monte-Carlo methods},
    author={Zafeirios Fountas and Sajid, Noor and Mediano, Pedro A.M. and Friston, Karl},
    journal={arXiv preprint arXiv:},
    year={2020}
  }
  ```
