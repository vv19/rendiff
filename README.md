# Render and Diffuse (R&D)

Code for paper: "Render and Diffuse: Aligning Image and Action Spaces for Diffusion-based Behaviour Cloning
" (RSS 2024).
[Project Webpage](https://vv19.github.io/render-and-diffuse/), [Paper](https://arxiv.org/pdf/2405.18196)

## Setup

**Clone this repo**

```
git clone https://github.com/vv19/rendiff.git
cd rendiff
```

**Create conda environment**

```
conda env create -f environment.yml
conda activate rendiff
```

Install PyRep and RLbench by following the instructions in the https://github.com/stepjam/PyRep
and https://github.com/stepjam/RLBench

```
pip install -e .
```

## Quick Start

### Try our pre-trained model for one of RLBench tasks.

Download pre-trained weights.

```
./scripts/download_weights.sh
```

Run inference. Set create_gifs to 1 if you want to visualise the rollout.

```
python3 -m rendiff.eval \
 --task_name='phone_on_base' \
 --run_name='POB' \
 --checkp_name='final' \
 --create_gifs=0
```

### Train your own model

First, record demonstrations for the task you are interested in, e.g.:

```
python3 -m rendiff.data_collection.record_demos \
 --task_name='lift_lid' \
 --datadir="/path/to/datadir" \
 --num_demos=100
```

Set your desired hyperparameters in the `rendif/configs/rendif_config.py` file.

Then, train the model:

```
python3 -m rendiff.train \
 --run_name='LIFT_LID' \
 --datadir='/path/to/datadir/task_name' \
 --num_demos=100
```

#### Performance

To reach the best performance on a given task different hyperparameters in `rendif_config.py` and arguments for `train.py` and `eval.py` should be tuned. 
We recommend saving multiple checkpoints during training and evaluating them to find the best model.
As with many Behaviour Cloning methods, more demonstrations and longer training times typically leads to better performance. 

## Some areas to improve upon

- [ ] Integrate gripper actions in a more elegant way.
- [ ] Find more optimal hyperparameters.
- [ ] Integrate pre-trained vision models.
- [ ] Data augmentation and regularization.
- [ ] Explore different model architectures.
- [ ] Optimise training and inference to reduce time computational requirements.

# Citing

If you find our paper interesting or this code useful in your work, please cite our paper:

```
@article{vosylius2024render,
  title={Render and Diffuse: Aligning Image and Action Spaces for Diffusion-based Behaviour Cloning},
  author={Vosylius, Vitalis and Seo, Younggyo and Uru{\c{c}}, Jafar and James, Stephen},
  journal={arXiv preprint arXiv:2405.18196},
  year={2024}
}
```
