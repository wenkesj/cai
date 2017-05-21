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

#include "matrix.h"
#include "criterion.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * criterion_create
 */
criterion *criterion_create(
  matrix *(*criterion_output)(matrix *, matrix *),
  matrix *(*criterion_gradient)(matrix *, matrix *)
) {
  criterion *c;
  if ((c = malloc(sizeof(*c))) == NULL) {
    perror("Out of memory\n");
    return NULL;
  }
  c->criterion_output = criterion_output;
  c->criterion_gradient = criterion_gradient;
  c->output = NULL;
  c->gradient = NULL;
  return c;
}

/*
 * criterion_forward calculates the 1d-loss
 */
matrix *criterion_forward(criterion *c, matrix *output, matrix *target) {
  if (c->output == NULL) {
    c->output = malloc(sizeof(*output));
  }
  *c->output = *c->criterion_output(output, target);
  return matrix_copy(c->output);
}

/*
 * criterion_backward calculates the gradient w.r.t the output
 */
matrix *criterion_backward(criterion *c, matrix *output, matrix *target) {
  if (c->gradient == NULL) {
    c->gradient = malloc(sizeof(*output));
  }
  *c->gradient = *c->criterion_gradient(output, target);
  return matrix_copy(c->gradient);
}

/*
 * criterion_forward_bce
 */
matrix *criterion_forward_bce(matrix *output, matrix *target) {
  int i, j;
  double sum = 0.0;
  matrix *c = matrix_create(1, 1, NULL);
  for (i = 0; i < output->rows; i++) {
    for (j = 0; j < output->columns; j++) {
      sum -= (float)((double)target->data[i][j] * log((double)output->data[i][j]) +
        (double)(1 - target->data[i][j]) * (double)log(1 - output->data[i][j]));
    }
  }
  c->data[0][0] = (float)(sum/output->rows);
  return c;
}

/*
 * criterion_backward_bce
 */
matrix *criterion_backward_bce(matrix *output, matrix *target) {
  int i, j;
  float norm = 1.0;
  matrix *gradient = matrix_create(output->rows, output->columns, NULL);
  for (i = 0; i < output->rows; i++) {
    for (j = 0; j < output->columns; j++) {
      gradient->data[i][j] = -norm * (target->data[i][j] - output->data[i][j]) /
        ((1.0 - output->data[i][j]) * output->data[i][j]);
    }
  }
  return gradient;
}

/*
 * criterion_forward_mse, calculate the mse loss
 */
matrix *criterion_forward_mse(matrix *output, matrix *target) {
  int i, j;
  float error = 0.0, sum = 0.0;
  matrix *c = matrix_create(1, 1, NULL);
  for (i = 0; i < output->rows; i++) {
    for (j = 0; j < output->columns; j++) {
      error = output->data[i][j] - target->data[i][j];
      sum += error * error;
    }
  }
  c->data[0][0] = sum / (float)output->rows;
  return c;
}

/*
 * criterion_backward_mse
 */
matrix *criterion_backward_mse(matrix *output, matrix *target) {
  int i, j;
  float norm = 2.0;
  matrix *gradient = matrix_create(output->rows, output->columns, NULL);
  for (i = 0; i < output->rows; i++) {
    for (j = 0; j < output->columns; j++) {
      gradient->data[i][j] = norm * (output->data[i][j] - target->data[i][j]);
    }
  }
  return gradient;
}

/*
 * criterion_free
 */
void criterion_free(criterion *c) {
  free(c->gradient);
  free(c->output);
  free(c);
  c = NULL;
}
