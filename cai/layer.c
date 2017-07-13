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
  l->biases = update == NULL ?
    NULL : matrix_create(output, 1, parameter_function == NULL ?
      &matrix_zeros : parameter_function);
  l->gradient_weights = update == NULL ?
    NULL : matrix_create(output, input, &matrix_zeros);
  l->gradient_biases = update == NULL ?
    NULL : matrix_create(output, 1, &matrix_zeros);
  l->update = update;
  return l;
}

/*
 * layer_forward
 */
matrix *layer_forward(layer *l, matrix *input) {
  *l->output = *l->forward(l, input);
  return matrix_copy(l->output);
}

/*
 * layer_backward
 */
matrix *layer_backward(layer *l, matrix *input, matrix *gradient) {
  *l->gradient = *l->backward(l, input, gradient);
  return matrix_copy(l->gradient);
}

/*
 * layer_update
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
  matrix *res = matrix_multiply(l->weights, input);
  matrix *output = matrix_add(res, l->biases);
  matrix_free(res);
  return output;
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
  matrix *gradient_biases = matrix_copy(l->gradient_biases);
  for (i = 0; i < l->gradient_weights->rows; i++) {
    for (j = 0; j < l->gradient_weights->columns; j++) {
      gradient_weights->data[i][j] += (scale * gradient->data[i][0] * input->data[j][0]);
    }
  }
  *l->gradient_biases = *matrix_add(l->gradient_biases, matrix_scale(gradient, scale));
  return gradient_weights;
}

/*
 * layer_forward_tanh
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
  return  ((2 * ((float)rand()/(float)(RAND_MAX))) - 1);
}

/*
 * layer_ones
 */
float layer_ones(int i, int j) {
  return 1.0;
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
