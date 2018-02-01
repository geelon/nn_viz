# Visualizing Neural Network Learning

The goal here is to develop some more intuition about how neural
networks learn: their dynamics and their limits. The specific
questions we ask here are inspired by certain interpretations on how
neural networks learn/generalize.

## Questions List

1. Do certain layers converge first? Perhaps earlier layers?
2. Layers as expanding/contracting uncertainty?
3. Variance of converged network? Are hidden layers comparable? Have
they extracted the same information?
4. If we bottleneck information through a layer of small dimension,
can we obtain the remaining useful information?

## Output

For example, here are two outputs following training on the iris
dataset:

<img src="output/validation_long.png" alt="train/test loss" />

<img src="output/validation_super_long.png" alt="super long train/test
loss" />

This was trained on a neural network with two hidden layers of
dimension 2 followed by 4.