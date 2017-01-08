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
    outputs = layer_forward(l, outputs);
  }
  return matrix_copy(((layer *)n->layers->tail->value)->output);
}

/*
 * network_backward
 */
matrix *network_backward(network *n, matrix *input, matrix *gradient) {
  list_node *layer_node;
  matrix *gradient_update = matrix_copy(gradient);
  matrix *input_update = matrix_copy(input);
  list_for_each_reverse (n->layers, layer_node) {
    layer *l = (layer *)layer_node->value;
    gradient_update = layer_backward(l, input_update, gradient_update);
    if (l->update != NULL) {
      layer_update(l, input_update, gradient_update, 1);
    }
    input_update = matrix_copy(l->output);
  }
  return matrix_copy(((layer *)n->layers->head->value)->gradient);
}

/*
 * network_update
 */
network *network_update(network *n, float learning_rate) {
  list_node *layer_node;
  list_for_each (n->layers, layer_node) {
    layer *l = (layer *)layer_node->value;
    if (l->weights && l->update) {
      int i, j;
      for (i = 0; i < l->weights->rows; i++) {
        for (j = 0; j < l->weights->columns; j++) {
          l->weights->data[i][j] += -learning_rate * l->gradient_weights->data[i][j];
        }
      }
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
    if (l->gradient_weights) {
      free(l->gradient_weights);
      l->gradient_weights = matrix_create(l->gradient_weights->rows, l->gradient_weights->columns, &matrix_zeros);
    }
    free(l->gradient);
    l->gradient = matrix_create(l->gradient->rows, l->gradient->columns, &matrix_zeros);
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
