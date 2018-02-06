# Experiment Notes

## Increasing Weights

**Hypothesis**: generalization error (in unregularizaed MLPs) is in
 part a result of increasing weights; as the size of the weights
 increases, multiplying by these weights expands the discernability of
 input data (i.e. 'increases the precision' of the input). 

**Simple Experimental Setup:**

- Iris dataset, (a) original, (b) transformed into confidence values
  on binary features.
- MLP, no regularization, Adam optimization
- Model architecture: [n input] -> [6 hidden] -> [4 hidden] -> [3
  output], where n = 4 for the original, and n = 12 for transformed.

### Original, L1-norm of weights

Here is the output for the original Iris dataset (no transformation
into binary features); architecture is [4 input] -> [6 hidden] -> [4
hidden] -> [3 output].

<img src="./output/l1_weights_original/loss.png" alt="Train
vs. Test loss"/>

<img src="./output/l1_weights_original/weights_evolution_zoom.png"
alt="Weights Evolution zoomed-in" />

### Original, L2-norm of weights

<img src="./output/l2_weights_original/loss.png" alt="Train vs. Test
loss"/>

<img src="./output/l2_weights_original/weights_evolution_zoom.png"
alt="Weights Evolution zoomed-in" />

### Transformed, L1-norm of weights

Here is the output for the transformed dataset. Architecture is [12
input] -> [6 hidden] -> [4 hidden] -> [3 output].

<img src="./output/l1_weights_transformed/loss.png" alt="Train vs. Test
loss"/>

<img src="./output/l1_weights_transformed/weights_evolution_zoom.png"
alt="Weights Evolution zoomed-in" />

### Transformed, L2-norm of weights

<img src="./output/l2_weights_transformed/loss.png" alt="Train vs. Test
loss"/>

<img src="./output/l2_weights_transformed/weights_evolution_zoom.png"
alt="Weights Evolution zoomed-in" />

## Generalization Error vs. Weights

Here are plots comparing generalization error, |test_error -
train_error|, and the weights. Here, the generalization error is on a
logarithmic scale while the weights are on a linear scale.*

<img
src="./output/generalization_v_weights/generalization_original_l1.png"
alt="Original l1 generalization error" />

<img
src="./output/generalization_v_weights/generalization_transformed_l1.png"
alt="Original l1 generalization error" />

<img
src="./output/generalization_v_weights/generalization_transformed_l2.png"
alt="Original l1 generalization error" />

*Specifically, we applied a transformation: exp(weights/8)/2700 to get
 the scales to be comparable.

**Unfortunately, I did not properly save the *original l2* data.