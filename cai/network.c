#include "layer.h"
#include "network.h"
#include "list.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * network_create
 */
network *network_create() {
  network *n;
  if ((n = malloc(sizeof(*n))) == NULL) {
    perror("Out of memory\n");
    return NULL;
  }
  n->layers = list_create();
  return n;
}

/*
 * network_layer_add
 */
network *network_layer_add(network *n, layer *l) {
  list_add(n->layers, (void *)l);
  return n;
}

/*
 * network_forward
 */
matrix *network_forward(network *n, matrix *input) {
  list_node *layer_node;
  matrix *outputs = matrix_copy(input);
  list_for_each (n->layers, layer_node) {
    layer *l = (layer *)layer_node->value;
    *outputs = *layer_forward(l, outputs);
  }
  return outputs;
}

/*
 * network_backward, two-pass
 */
matrix *network_backward(network *n, matrix *input, matrix *gradient) {
  list_node *layer_node;
  matrix *gradient_update = matrix_copy(gradient);
  matrix *output;

  // Update gradients
  list_for_each_reverse (n->layers, layer_node) {
    layer *l = (layer *)layer_node->value;
    if (layer_node->previous != NULL) {
      layer *pl = (layer *)layer_node->previous->value;
      output = pl->output;
    } else {
      output = input;
    }
    *gradient_update = *layer_backward(l, output, gradient_update);
  }
  *gradient_update = *matrix_copy(gradient);

  // Update gradient weights
  list_for_each_reverse (n->layers, layer_node) {
    layer *l = (layer *)layer_node->value;
    if (layer_node->previous != NULL) {
      layer *pl = (layer *)layer_node->previous->value;
      output = pl->output;
    } else {
      output = input;
    }

    if (l->update != NULL) {
      layer_update(l, output, gradient_update, 1);
    }

    *gradient_update = *matrix_copy(l->gradient);
  }

  return gradient_update;
}

/*
 * network_update
 */
network *network_update(network *n, float learning_rate) {
  list_node *layer_node;
  list_for_each (n->layers, layer_node) {
    layer *l = (layer *)layer_node->value;
    if (l->weights != NULL) {
      matrix *scaled_weights = matrix_scale(l->gradient_weights, -learning_rate);
      *l->weights = *matrix_add(l->weights, scaled_weights);
      matrix_free(scaled_weights);
    }
    if (l->biases != NULL) {
      matrix *scaled_biases = matrix_scale(l->gradient_biases, -learning_rate);
      *l->biases = *matrix_add(l->biases, scaled_biases);
      matrix_free(scaled_biases);
    }
  }
  return n;
}

/*
 * network_gradient_zero
 */
void network_gradient_zero(network *n) {
  list_node *layer_node;
  list_for_each (n->layers, layer_node) {
    layer *l = (layer *)layer_node->value;
    if (l->gradient_weights != NULL) {
      *l->gradient_weights = *matrix_create(
        l->gradient_weights->rows,
        l->gradient_weights->columns,
        &matrix_zeros
      );
    }
    if (l->gradient_biases != NULL) {
      *l->gradient_biases = *matrix_create(
        l->gradient_biases->rows,
        l->gradient_biases->columns,
        &matrix_zeros
      );
    }
  }
}

/*
 * network_free
 */
void network_free(network *n) {
  list_node *layer_node;
  list_for_each (n->layers, layer_node) {
    layer_free((layer *)layer_node->value);
  }
  list_free(n->layers);
  free(n);
  n = NULL;
}
