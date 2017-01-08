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
#include <stdlib.h>
#include <stdio.h>

/*
 * matrix_zeros
 */
float matrix_zeros(int i, int j) {
  return 0.0;
}

/*
 * matrix_ones
 */
float matrix_ones(int i, int j) {
  return 1.0;
}

/*
 * matrix_create
 */
matrix *matrix_create(int rows, int columns, float (*initialize_function)(int, int)) {
  matrix *m;
  if ((m = malloc(sizeof(*m))) == NULL) {
    perror("Out of memory\n");
    return NULL;
  }
  int i, j;
  m->rows = rows;
  m->columns = columns;
  initialize_function = initialize_function != NULL ? initialize_function : &matrix_zeros;

  float **data = (float **)malloc(rows * sizeof(float *));

  for (i = 0; i < rows; i++) {
    data[i] = (float *)malloc(columns * sizeof(float));
    for (j = 0; j < columns; j++) {
      data[i][j] = initialize_function(i, j);
    }
  }

  m->data = data;
  return m;
}

/*
 * matrix_multiply
 */
matrix *matrix_multiply(matrix *a, matrix *b) {
  int i, j, k;
  matrix *c = matrix_create(a->rows, b->columns, &matrix_zeros);
  matrix *bT = matrix_transpose(b);
	for (i = 0; i < a->rows; i++) {
		for (j = 0; j < b->columns; j++) {
      for (k = 0; k < a->columns; k++) {
        c->data[i][j] += (a->data[i][k] * bT->data[j][k]);
      }
    }
  }
  matrix_free(bT);
  return c;
}

/*
 * matrix_add
 */
matrix *matrix_add(matrix *a, matrix *b) {
  int i, j;
  matrix *c = matrix_create(a->rows, a->columns, NULL);
  for (i = 0; i < a->rows; i++) {
    for (j = 0; j < a->columns; j++) {
      c->data[i][j] = a->data[i][j] + b->data[i][j];
    }
  }
  return c;
}

/*
 * matrix_scale
 */
matrix *matrix_scale(matrix *a, float b) {
  int i, j;
  matrix *c = matrix_create(a->rows, a->columns, NULL);
  for (i = 0; i < a->rows; i++) {
    for (j = 0; j < a->columns; j++) {
      c->data[i][j] = a->data[i][j] * b;
    }
  }
  return c;
}

/*
 * matrix_transpose
 */
matrix *matrix_transpose(matrix *m) {
  int i, j;
  matrix *c = matrix_create(m->columns, m->rows, NULL);
  for (i = 0; i < m->rows; i++) {
    for (j = 0; j < m->columns; j++) {
      c->data[j][i] = m->data[i][j];
    }
  }
  return c;
}

/*
 * matrix_copy
 */
matrix *matrix_copy(matrix *m) {
  matrix *copy;
  if ((copy = malloc(sizeof(*copy))) == NULL) {
    perror("Out of memory\n");
    return NULL;
  }
  int i, j;
  copy->rows = m->rows;
  copy->columns = m->columns;
  copy->data = (float **)malloc(copy->rows * sizeof(float *));

  for (i = 0; i < copy->rows; i++) {
    copy->data[i] = (float *)malloc(copy->columns * sizeof(float));
    for (j = 0; j < copy->columns; j++) {
      copy->data[i][j] = m->data[i][j];
    }
  }
  return copy;
}

/*
 * matrix_string
 */
char *matrix_string(matrix *m) {
  int i, j;
  char *buffer = malloc(2 * m->columns * m->rows * sizeof(char));
  for (i = 0; i < m->rows; i++) {
    for (j = 0; j < m->columns; j++) {
      sprintf(buffer, j != m->columns - 1 ? "%f\t" : "%f\n", m->data[i][j]);
    }
  }
  return buffer;
}

/*
 * matrix_free
 */
void matrix_free(matrix *m) {
  if (m != NULL) {
    free(m->data);
    free(m);
    m = NULL;
  }
}