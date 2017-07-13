#ifndef __NETWORK_H__
#define __NETWORK_H__
#include "list.h"
#include "matrix.h"
#include "layer.h"

typedef struct network {
  list *layers;
} network;

network *network_create();
network *network_layer_add(network *n, layer *l);
matrix *network_forward(network *n, matrix *input);
matrix *network_backward(network *n, matrix *output, matrix *gradient);
network *network_update(network *n, float learning_rate);
void network_gradient_zero(network *n);
void network_free(network *n);

#endif
