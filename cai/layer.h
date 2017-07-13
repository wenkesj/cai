#ifndef __LAYER_H__
#define __LAYER_H__
#include "matrix.h"

typedef struct layer {
  matrix *(*forward)(struct layer *l, matrix *);
  matrix *(*backward)(struct layer *l, matrix *, matrix *);
  matrix *(*update)(struct layer *l, matrix *input, matrix *gradient, float learning_rate);
  matrix *weights;
  matrix *biases;
  matrix *output;
  matrix *gradient;
  matrix *gradient_weights;
  matrix *gradient_biases;
} layer;

layer *layer_create(
  matrix *(*forward)(layer *l, matrix *),
  matrix *(*backward)(layer *l, matrix *, matrix *),
  matrix *(*update)(layer *l, matrix *input, matrix *gradient, float learning_rate),
  float (*parameter_function)(int, int),
  int input,
  int output
);
matrix *layer_forward(layer *l, matrix *input);
matrix *layer_backward(layer *l, matrix *output, matrix *gradient);
matrix *layer_update(layer *l, matrix *output, matrix *gradient, float learning_rate);
matrix *layer_forward_sigmoid(layer *l, matrix *output);
matrix *layer_backward_sigmoid(layer *l, matrix *output, matrix *gradient);
matrix *layer_forward_linear(layer *l, matrix *output);
matrix *layer_backward_linear(layer *l, matrix *output, matrix *gradient);
matrix *layer_update_linear(layer *l, matrix *input, matrix *gradient, float learning_rate);
matrix *layer_forward_tanh(layer *l, matrix *output);
matrix *layer_backward_tanh(layer *l, matrix *output, matrix *gradient);
matrix *layer_forward_none(layer *l, matrix *output);
matrix *layer_backward_none(layer *l, matrix *output, matrix *gradient);
float layer_random(int i, int j);
float layer_ones(int i, int j);
void layer_free(layer *l);

#endif
