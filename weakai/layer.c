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
  l->output = matrix_create(output, 1, NULL);
  l->gradient = matrix_create(output, 1, NULL);
  l->weights = update == NULL ?
    NULL : matrix_create(output, input, parameter_function == NULL ?
      &matrix_zeros : parameter_function);
  l->gradient_weights = update == NULL ?
    NULL : matrix_create(output, input, &matrix_zeros);
  l->update = update;
  return l;
}

/*
 * layer_forward, f(x)
 */
matrix *layer_forward(layer *l, matrix *input) {
  *l->output = *l->forward(l, input);
  return matrix_copy(l->output);
}

/*
 * layer_backward, δ = δE * f'(x)
 */
matrix *layer_backward(layer *l, matrix *input, matrix *gradient) {
  *l->gradient = *l->backward(l, input, gradient);
  return matrix_copy(l->gradient);
}

/*
 * layer_update,
 */
matrix *layer_update(layer *l, matrix *input, matrix *gradient, float scale) {
  *l->gradient_weights = *l->update(l, input, gradient, scale);
  return matrix_copy(l->gradient_weights);
}

/*
 * layer_forward_sigmoid
 */
matrix *layer_forward_sigmoid(layer *l, matrix *input) {
  int i, j;
  matrix *output = matrix_create(input->rows, input->columns, NULL);
  for (i = 0; i < input->rows; i++) {
    for (j = 0; j < input->columns; j++) {
      output->data[i][j] = 1 / (1 + exp(-input->data[i][j]));
    }
  }
  return output;
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
  return matrix_multiply(l->weights, input);
}

/*
 * layer_backward_linear
 */
matrix *layer_backward_linear(layer *l, matrix *output, matrix *gradient) {
  matrix *wt = matrix_transpose(l->weights);
  matrix *g = matrix_multiply(wt, gradient);
  matrix_free(wt);
  return g;
}

/*
 * layer_update_linear
 */
matrix *layer_update_linear(layer *l, matrix *input, matrix *gradient, float scale) {
  int i, j;
  matrix *gradient_weights = matrix_copy(l->gradient_weights);
  for (i = 0; i < l->gradient_weights->rows; i++) {
    for (j = 0; j < l->gradient_weights->columns; j++) {
      gradient_weights->data[i][j] += (scale * input->data[j][0] * gradient->data[i][0]);
    }
  }
  return gradient_weights;
}

/*
 * layer_forward_tanh, x[i] = f(x[i-1])
 */
matrix *layer_forward_tanh(layer *l, matrix *input) {
  int i, j;
  matrix *layer_output = matrix_create(input->rows, input->columns, NULL);
  for (i = 0; i < layer_output->rows; i++) {
    for (j = 0; j < layer_output->columns; j++) {
      layer_output->data[i][j] = (float)tanh((double)input->data[i][j]);
    }
  }
  return layer_output;
}

/*
 * layer_backward_tanh, f'(o)
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
  return (2 * ((float)rand()/(float)(RAND_MAX))) - 1;
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
