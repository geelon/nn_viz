# Experiment Notes

## Increasing Weights

Here is training run on the Iris dataset (transformed into confidence
values on binary features), with no regularization, Adam optimizer,
and with model architecture [12 input] -> [6 hidden] -> [4 hidden] ->
[3 output]. Here is the loss:

<img src="./output/weights_increasing/loss.png" alt="Train vs. Test
loss"/>

Here is a l1-norm of the weights for the connections between each
layer:

<img src="./output/weights_increasing/weights_evolution_zoom.png"
alt="Weights Evolution zoomed-in" />

Here is the output for the original Iris dataset (no transformation
into binary features); architecture is [12 input] -> [6 hidden] -> [4
hidden] -> [3 output].

<img src="./output/weights_increasing_original/loss.png" alt="Train
vs. Test loss"/>

<img src="./output/weights_increasing_original/weights_evolution_zoom.png"
alt="Weights Evolution zoomed-in" />

And finally, here is the output, same as before, except with L2 norm.

<img src="./output/l2_weights_original/loss.png" alt="Train vs. Test
loss"/>

<img src="./output/l2_weights_original/weights_evolution_zoom.png"
alt="Weights Evolution zoomed-in" />