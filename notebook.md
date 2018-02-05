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

<img src="./output/weights_increasing/weights_evolution.png"
alt="Weights Evolution"/>

<img src="./output/weights_increasing/weights_evolution_zoom.png"
alt="Weights Evolution zoomed-in" />