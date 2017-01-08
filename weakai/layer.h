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

#ifndef __LAYER_H__
#define __LAYER_H__
#include "matrix.h"

typedef struct layer {
  matrix *(*forward)(struct layer *l, matrix *);
  matrix *(*backward)(struct layer *l, matrix *, matrix *);
  matrix *(*update)(struct layer *l, matrix *input, matrix *gradient, float learning_rate);
  matrix *weights;
  matrix *output;
  matrix *gradient;
  matrix *gradient_weights;
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
void layer_free(layer *l);

#endif
