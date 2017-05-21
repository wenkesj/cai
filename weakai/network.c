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
 *   * Neither the name of Sam Wenke nor the names of its contributors may be
 * used
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

  matrix *outputs_copy = matrix_copy(outputs);
  matrix_free(outputs);
  return outputs_copy;
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

    // previous layers output as input
    *gradient_update = *layer_backward(l, output, gradient_update);
  }
  *gradient_update = *gradient;

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
    *gradient_update = *l->gradient;
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
      matrix_print("w", l->gradient_weights);
      matrix *scaled_weights = matrix_scale(l->gradient_weights, -learning_rate);
      matrix_print("sw", scaled_weights);
      *l->weights = *matrix_add(l->weights, scaled_weights);
      matrix_free(scaled_weights);
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
      *l->gradient_weights = *matrix_create(l->gradient_weights->rows, l->gradient_weights->columns, &matrix_zeros);
    }
    if (l->gradient != NULL) {
      *l->gradient = *matrix_create(l->gradient->rows, l->gradient->columns, &matrix_zeros);
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
