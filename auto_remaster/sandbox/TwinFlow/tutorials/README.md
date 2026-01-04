## Tutorials

In the `tutorials/mnist` directory, we provide training code for MNIST that closely follows the implementation described in the paper, intended for tutorial purposes. This tutorial includes the core implementations of $L_\mathrm{base}$ and $\mathcal{L}_\mathrm{TwinFlow}$ .

To run TwinFlow training:

```bash
cd tutorials/mnist
python main.py --using_twinflow --save_dir ./outputs/twinflow
```

To run training without $\mathcal{L}_\mathrm{TwinFlow}$:

```bash
cd tutorials/mnist
python main.py --save_dir ./outputs/rcgm
```

| TwinFlow training on MNIST (1-NFE) | RCGM (without TwinFlow) training on MNIST (1-NFE) |
|----------------------------|------------------------------------------|
| ![](../assets/mnist_twinflow.png) | ![](../assets/mnist_rcgm.png) |