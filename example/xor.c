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

#include "../lib/matrix.h"
#include "../lib/network.h"
#include "../lib/layer.h"
#include "../lib/criterion.h"
#include "../lib/sgd.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  srand(time(NULL));
  int input_dimensions = 2;
  int output_dimensions = 1;
  int hidden_dimensions = 10;
  int training_iterations = 100;
  int testing_iterations = 10;
  float learning_rate = 1e-5;

  network *n = network_create();

  sgd *s = sgd_create(learning_rate, 0, 0, 0, 0, 0);

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

  // Train
  int i, j;
  matrix *input, *target, *output, *loss, *gradient;
  for (i = 0; i < training_iterations; i++) {
    // Create an XOR input and target on the fly
    input = matrix_create(1, input_dimensions, NULL);
    target = matrix_create(1, output_dimensions, NULL);
    input->data[0][0] = (2 * ((float)rand()/(float)(RAND_MAX))) - 1;
    input->data[0][1] = (2 * ((float)rand()/(float)(RAND_MAX))) - 1;
    target->data[0][0] = (input->data[0][0] * input->data[0][1] > 0) ? -1 : 1;

    // Reset gradients
    network_gradient_zero(n);

    // Forward pass
    output = network_forward(n, input);
    loss = criterion_forward(c, output, target);

    // Backward pass
    gradient = criterion_backward(c, output, target);
    network_backward(n, input, gradient);

    // Evaluate and update network by stochasitc gradient decent
    sgd_evaluate(s, n);

    // Free up some extra space
    free(input);
    free(target);
    free(output);
    free(loss);
    free(gradient);
  }

  // Test
  int k, l;
  int total_correct = 0;
  for (i = 0; i < testing_iterations; i++) {
    input = matrix_create(1, input_dimensions, NULL);
    target = matrix_create(1, output_dimensions, NULL);
    input->data[0][0] = (2 * ((float)rand()/(float)(RAND_MAX))) - 1;
    input->data[0][1] = (2 * ((float)rand()/(float)(RAND_MAX))) - 1;
    target->data[0][0] = (input->data[0][0] * input->data[0][1] > 0) ? -1 : 1;

    output = network_forward(n, input);
    printf("%s %f %f\n", "Output", output->data[0][0], target->data[0][0]);

    total_correct += ((output->data[0][0] > 0) ? 1 : -1) == target->data[0][0] ? 1 : 0;

    free(output);
    free(input);
    free(target);
  }
  printf("%s %.2f %%\n", "Percent Correct", ((float)total_correct / (float)testing_iterations) * 100.0);

  // Clean up!
  criterion_free(c);
  network_free(n);
}
