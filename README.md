# Visualizing Neural Network Learning

The goal here is to develop some more intuition about how neural
networks learn: their dynamics and their limits. The specific
questions we ask here are inspired by certain interpretations on how
neural networks learn/generalize.

## Real Features vs. Artifacts

If we take the perspective that neurons in a neural network transmit
computations to the next layer, then reasonably, we'd want the
computation to convey something 'real' about in the data, and not some
artifact of the data. 

One possible criterion for the 'realness' of a feature is whether
there are many experiments we could perform to reveal that
feature. In other words, is there broad evidence for that feature?

We could interpret dropout technique as asking this question: if I
obscure some of the data, can the feature still be computed?
Analogously, in real life, humans understand blurry pictures or deduce
from partial information. 

Perhaps the usual neural network (MLP) structure computes this
question. Though we usually think the output of each neuron as the
result of a computation, maybe it is closer to the confidence of that
computation. Thus, each neuron represents a computation, but from the
input data, the actual computation being run by the computer, usually
something like σ(W*x + b), determines how certain the computation
would be.

Furthermore, this gives the interpretation of regularization as
preventing the model from becoming overly dependent on a small subset
of data.

This might help ensure low generalization error, as the model is then
forced to learn 'real' features.

### Experimentation

Note that this will focus mostly on MLPs, as it seems that other
techniques that build onto MLPs mostly incorporate data-dependent
optimizations (e.g. CNNs assume translation invariance).

1. How can we constrain/modify the MLP according to the above
interpretation?
2. Could we have each neuron pass a pair,
```(computation,confidence)``` instead of just ```confidence```? Or
perhaps we could work with Boolean features instead (in which case
we'd just need a ```confidence``` value).
3. Can we recode real values into binary values that incorporate
uncertainty, thus propagating uncertainty?



## Unstructured Questions List

1. Do certain layers converge first? Perhaps earlier layers?
2. Layers as expanding/contracting uncertainty?
3. Variance of converged network? Are hidden layers comparable? Have
they extracted the same information?
4. If we bottleneck information through a layer of small dimension,
can we obtain the remaining useful information?
5. Does decreasing the batch size over time mimic increasing
discernment? 

## Output

For example, here are two outputs following training on the iris
dataset:

<img src="output/validation_long.png" alt="train/test loss" />

<img src="output/validation_super_long.png" alt="super long train/test
loss" />

The x-axis is the training epoch, while the y-axis is the training and
testing loss. This was trained on a neural network with two hidden
layers of dimension 2 followed by 4.