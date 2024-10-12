# JAX Diffusion

This is a toy project that implements a [Denoising Diffusion Probabilistic Model (DDPM)](https://arxiv.org/pdf/2006.11239) in JAX, using the MNIST dataset.

## Setup

Getting JAX running with the Metal backend (for macOS) can be a bit challenging. The `requirements.txt` file is tailored to my setup, so you may need to adjust the dependencies to suit your environment. I suggest unpinning the dependencies in `requirements.txt` to start with.

## Commands

For model training:

```bash
python3 train.py --expdir $EXPDIR --epochs 20 
```

To run inference:

```bash
python3 ddpm.py --checkpoint $CHECKPOINT_PATH --num_images $NUM_IMAGES
```

## Example Generations

After around 15 epochs of training on an Apple M2 (which takes a few hours), the model begins generating fairly convincing samples:

![](/assets/output_3.jpg)
![](/assets/output_4.jpg)
![](/assets/output_5.jpg)
![](/assets/output_8.jpg)
![](/assets/output_9.jpg)

![](/assets/output_3.gif)
![](/assets/output_4.gif)
![](/assets/output_5.gif)
![](/assets/output_8.gif)
![](/assets/output_9.gif)
