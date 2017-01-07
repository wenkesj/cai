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

#include "sgd.h"
#include "matrix.h"
#include "network.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * sgd_create
 */
sgd *sgd_create(
  float learning_rate,
  float learning_rate_decay,
  float weight_decay,
  float momentum,
  float dampening,
  float evaluations
) {
  sgd *s;
  if ((s = malloc(sizeof(*s))) == NULL) {
    perror("Out of memory\n");
    return NULL;
  }
  s->learning_rate = learning_rate ? learning_rate : 1e-3;
  s->learning_rate_decay = learning_rate_decay ? learning_rate_decay : 0;
  s->weight_decay = weight_decay ? weight_decay : 0;
  s->momentum = momentum ? momentum : 0;
  s->dampening = dampening ? dampening : s->momentum;
  s->evaluations = evaluations ? evaluations : 0;
  return s;
}

/*
 * sgd_evaluate
 */
sgd *sgd_evaluate(sgd *s, network *n) {
  float learning_rate = s->learning_rate /
    (1 + (s->evaluations * s->learning_rate_decay));

  // Iterate over all parameters
  list_node *list_layer;
  list_for_each (n->layers, list_layer) {
    layer *l = (layer *)list_layer->value;

    if (l->weights) {
      // Apply weight decay
      // dw/dx = dw/dx + w * dw/dt
      if (s->weight_decay != 0) {
        l->gradient_weights = matrix_add(l->gradient_weights,
          matrix_scale(l->weights, s->weight_decay));
      }

      // Apply momentum
      // dw/dx = dw/dx + (1 - d) * (w * m)
      if (s->momentum != 0) {
        l->gradient_weights = matrix_add(l->gradient_weights,
          matrix_scale(
            matrix_scale(l->gradient_weights, s->momentum),
              1 - s->dampening));
      }

      // Apply learning rate
      // w = w - dw/dx * lr
      l->weights = matrix_add(l->weights,
        matrix_scale(l->gradient_weights, -learning_rate));
    }
  }

  s->evaluations++;
  return s;
}

/*
 * sgd_free
 */
void sgd_free(sgd *s) {
  free(s);
}
