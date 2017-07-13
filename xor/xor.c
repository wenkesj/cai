#include <cai/criterion.h>
#include <cai/matrix.h>
#include <cai/layer.h>
#include <cai/network.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

float uniform() {
  return (2 * ((float)rand()/(float)(RAND_MAX))) - 1;
}

int main(int argc, char **argv) {
  srand(time(NULL));
  int input_dimensions = 2;
  int output_dimensions = 1;
  int hidden_dimensions = 20;
  int testing_iterations = 1000;
  int training_iterations = 2.0e3;

  network *n = network_create();

  // MSE criterion
  criterion *c = criterion_create(
    &criterion_forward_mse,
    &criterion_backward_mse
  );

  // Linear
  network_layer_add(n,
    layer_create(
      &layer_forward_linear,
      &layer_backward_linear,
      &layer_update_linear,
      &layer_random,
      input_dimensions,
      hidden_dimensions
    )
  );

  // Tanh
  network_layer_add(n,
    layer_create(
      &layer_forward_tanh,
      &layer_backward_tanh,
      NULL,
      NULL,
      hidden_dimensions,
      hidden_dimensions
    )
  );

  // Linear
  network_layer_add(n,
    layer_create(
      &layer_forward_linear,
      &layer_backward_linear,
      &layer_update_linear,
      &layer_random,
      hidden_dimensions,
      output_dimensions
    )
  );

  // Train, no validation
  matrix *input, *target, *output, *loss, *gradient;
  input = matrix_create(input_dimensions, 1, NULL);
  target = matrix_create(output_dimensions, 1, NULL);

  int epoch, correct = 0;
  for (epoch = 0; epoch < training_iterations; epoch++) {
    // Create an XOR input and target on the fly
    input->data[0][0] = uniform();
    input->data[1][0] = uniform();
    target->data[0][0] = (input->data[0][0] * input->data[1][0] > 0) ? -1 : 1;

    // Forward pass
    output = network_forward(n, input);
    loss = criterion_forward(c, output, target);
    gradient = criterion_backward(c, output, target);

    network_gradient_zero(n);
    network_backward(n, input, gradient);
    network_update(n, 0.001);
  }

  // Free up some extra space, the gradient return and loss are meta we won't really need.
  matrix_free(loss);
  matrix_free(gradient);

  // Test
  int total_correct = 0;
  for (epoch = 0; epoch < testing_iterations; epoch++) {
    input->data[0][0] = uniform();
    input->data[1][0] = uniform();
    target->data[0][0] = (input->data[0][0] * input->data[1][0] > 0) ? -1 : 1;
    output = network_forward(n, input);
    total_correct += (output->data[0][0] > 0 ? 1 : -1) == target->data[0][0] ? 1 : 0;
  }

  // Clean up everything
  matrix_free(output);
  matrix_free(input);
  matrix_free(target);
  criterion_free(c);
  network_free(n);

  // Report the results
  printf(
    "%s %.2f %%\n",
    "Percent Correct",
    ((float)total_correct / (float)testing_iterations) * 100.0
  );
}
