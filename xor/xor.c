/*
 * Copyright (c) 2016-2017, Sam Wenke <samwenke at gmail dot com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of Sam Wenke nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without
 *     specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <weakai/matrix.h>
#include <weakai/network.h>
#include <weakai/layer.h>
#include <weakai/criterion.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// random [-1, 1]
float uniform() {
  return (2 * ((float)rand()/(float)(RAND_MAX))) - 1 > 1 ? 1 : -1;
}

int main(int argc, char **argv) {
  srand(time(NULL));
  int input_dimensions = 2;
  int output_dimensions = 1;
  int hidden_dimensions = 10;
  int testing_iterations = 1e2;
  int training_iterations = 2e3;

  network *n = network_create();

  // MSE criterion
  criterion *c = criterion_create(
    &criterion_forward_mse,
    &criterion_backward_mse);

  // Linear
  network_layer_add(n,
    layer_create(
      &layer_forward_linear,
      &layer_backward_linear,
      &layer_update_linear,
      &layer_random,
      input_dimensions,
      hidden_dimensions));

  // Tanh
  network_layer_add(n,
    layer_create(
      &layer_forward_tanh,
      &layer_backward_tanh,
      NULL,
      NULL,
      hidden_dimensions,
      hidden_dimensions));

  // Linear
  network_layer_add(n,
    layer_create(
      &layer_forward_linear,
      &layer_backward_linear,
      &layer_update_linear,
      &layer_random,
      hidden_dimensions,
      output_dimensions));

  // Train, no validation
  matrix *input, *target, *output, *loss, *gradient;
  input = matrix_create(input_dimensions, 1, NULL);
  target = matrix_create(output_dimensions, 1, NULL);

  int epoch;
  for (epoch = 0; epoch < training_iterations; epoch++) {
    // Create an XOR input and target on the fly
    input->data[0][0] = uniform();
    input->data[1][0] = uniform();
    target->data[0][0] = (input->data[0][0] * input->data[1][0] > 0) ? -1 : 1;

    // Forward pass
    output = network_forward(n, input);
    loss = criterion_forward(c, output, target);

    // Reset gradients
    network_gradient_zero(n);
    gradient = criterion_backward(c, output, target);
    network_backward(n, input, gradient);
    network_update(n, 1e-3);
  }

  // Free up some extra space, the gradient return and loss are meta
  matrix_free(loss);
  matrix_free(gradient);

  // Test
  int total_correct = 0;
  for (epoch = 0; epoch < testing_iterations; epoch++) {
    input->data[0][0] = uniform();
    input->data[1][0] = uniform();
    target->data[0][0] = (input->data[0][0] * input->data[1][0] > 0) ? -1 : 1;
    printf("t %f\n", target->data[0][0]);
    printf("i %f %f\n", input->data[0][0], input->data[1][0]);
    output = network_forward(n, input);
    total_correct += (output->data[0][0] > 0 ? 1 : -1) == target->data[0][0] ? 1 : 0;
  }

  // Clean up everything and report the results
  matrix_free(output);
  matrix_free(input);
  matrix_free(target);
  criterion_free(c);
  network_free(n);

  printf("%s %.2f %%\n", "Percent Correct", ((float)total_correct / (float)testing_iterations) * 100.0);
}
