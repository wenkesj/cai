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

#include "layer.h"
#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * layer_create
 */
layer *layer_create(
  matrix *(*forward)(layer *, matrix *),
  matrix *(*backward)(layer *, matrix *, matrix *),
  matrix *(*update)(layer *, matrix *, matrix *, float),
  float (*parameter_function)(int, int),
  int input,
  int output
) {
  layer *l;
  if ((l = malloc(sizeof(*l))) == NULL) {
    perror("Out of memory\n");
    return NULL;
  }
  l->forward = forward == NULL ?
    &layer_forward_none : forward;
  l->backward = backward == NULL ?
    &layer_backward_none : backward;
  l->output = matrix_create(1, output, NULL);
  l->gradient = matrix_create(1, output, NULL);
  l->weights = update == NULL ?
    NULL : matrix_create(input, output, parameter_function == NULL ?
      &matrix_zeros : parameter_function);
  l->gradient_weights = update == NULL ?
    NULL : matrix_create(input, output, &matrix_zeros);
  l->update = update;
  return l;
}

/*
 * layer_forward
 */
matrix *layer_forward(layer *l, matrix *m) {
  l->output = l->forward(l, m);
  return matrix_copy(l->output);
}

/*
 * layer_backward
 */
matrix *layer_backward(layer *l, matrix *input, matrix *gradient) {
  l->gradient = l->backward(l, input, gradient);
  return matrix_copy(l->gradient);
}

/*
 * layer_update
 */
matrix *layer_update(layer *l, matrix *input, matrix *gradient, float learning_rate) {
  l->gradient_weights = l->update(l, input, gradient, learning_rate);
  return matrix_copy(l->gradient_weights);
}

/*
 * layer_forward_sigmoid
 */
matrix *layer_forward_sigmoid(layer *l, matrix *input) {
  int i, j;
  for (i = 0; i < input->rows; i++) {
    for (j = 0; j < input->columns; j++) {
      input->data[i][j] = 1 / (1 + exp(-input->data[i][j]));
    }
  }
  return input;
}

/*
 * layer_backward_sigmoid
 */
matrix *layer_backward_sigmoid(layer *l, matrix *output, matrix *gradient) {
  int i, j;
  matrix *gradient_update = matrix_create(output->rows, output->columns, NULL);
  for (i = 0; i < output->rows; i++) {
    for (j = 0; j < output->columns; j++) {
      gradient_update->data[i][j] = gradient->data[i][j] * output->data[i][j] * (1 - output->data[i][j]);
    }
  };
  return gradient_update;
}

/*
 * layer_forward_linear
 */
matrix *layer_forward_linear(layer *l, matrix *input) {
  return matrix_multiply(input, matrix_transpose(l->weights));
}

/*
 * layer_backward_linear
 */
matrix *layer_backward_linear(layer *l, matrix *output, matrix *gradient) {
  return matrix_multiply(gradient, l->weights);
}

/*
 * layer_update_linear
 */
matrix *layer_update_linear(layer *l, matrix *input, matrix *gradient, float learning_rate) {
  int i, j;
  matrix *gradient_weights = matrix_copy(l->gradient_weights);
  if (gradient_weights->rows != input->columns ||
      gradient_weights->columns != gradient->columns) {
    perror("Gradient/input dimensions don't match weights\n");
    return NULL;
  }
  for (i = 0; i < l->gradient_weights->rows; i++) {
    for (j = 0; j < l->gradient_weights->columns; j++) {
      gradient_weights->data[i][j] += (learning_rate * input->data[0][j] * gradient->data[0][i]);
    }
  }
  return gradient_weights;
}

/*
 * layer_forward_tanh
 */
matrix *layer_forward_tanh(layer *l, matrix *output) {
  int i, j;
  matrix *layer_output = matrix_copy(output);
  for (i = 0; i < layer_output->rows; i++) {
    for (j = 0; j < layer_output->columns; j++) {
      layer_output->data[i][j] = (float)tanh((double)layer_output->data[i][j]);
    }
  }
  return layer_output;
}

/*
 * layer_backward_tanh
 */
matrix *layer_backward_tanh(layer *l, matrix *output, matrix *gradient) {
  int i, j;
  matrix *gradient_update = matrix_create(output->rows, output->columns, NULL);
  for (i = 0; i < gradient_update->rows; i++) {
    for (j = 0; j < gradient_update->columns; j++) {
      gradient_update->data[i][j] = gradient->data[i][j] * (1 - output->data[i][j] * output->data[i][j]);
    }
  }
  return gradient_update;
}

/*
 * layer_forward_none
 */
matrix *layer_forward_none(layer *l, matrix *output) {
  return matrix_copy(output);
}

/*
 * layer_backward_none
 */
matrix *layer_backward_none(layer *l, matrix *output, matrix *gradient) {
  return matrix_copy(gradient);
}

/*
 * layer_random
 */
float layer_random(int i, int j) {
  return (float)rand()/(float)(RAND_MAX);
}

/*
 * layer_free
 */
void layer_free(layer *l) {
  if (l->output != NULL) {
    matrix_free(l->output);
    l->output = NULL;
  }
  if (l->weights != NULL) {
    matrix_free(l->weights);
    l->weights = NULL;
  }
  if (l->gradient_weights != NULL) {
    matrix_free(l->gradient_weights);
    l->gradient_weights = NULL;
  }
  if (l->gradient != NULL) {
    matrix_free(l->gradient);
    l->gradient = NULL;
  }
  free(l);
  l = NULL;
}
