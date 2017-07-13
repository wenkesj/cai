#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

/*
 * matrix_create returns a new matrix.
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
 * matrix_multiply returns a product matrix of a and b.
 */
matrix *matrix_multiply(matrix *a, matrix *b) {
  int i, j, k;
  matrix *c = matrix_create(a->rows, b->columns, &matrix_zeros);
  matrix *bT = matrix_transpose(b);
	for (i = 0; i < a->rows; i++) {
		for (j = 0; j < b->columns; j++) {
      c->data[i][j] = 0;
      for (k = 0; k < a->columns; k++) {
        c->data[i][j] += (a->data[i][k] * bT->data[j][k]);
      }
    }
  }
  matrix_free(bT);
  return c;
}

/*
 * matrix_add returns a additive matrix of a and b.
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
 * matrix_scale returns a scaled matrix of a.
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
 * matrix_transpose returns a transpose matrix of m.
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
 * matrix_copy returns a deep copy of m.
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
 * matrix_print prints matrix to stdout, lazy
 */
void matrix_print(char *name, matrix *input) {
  int i, j;
  printf("%s (%d, %d)\n", name, input->rows, input->columns);
  for (i = 0; i < input->rows; i++) {
    for (j = 0; j < input->columns; j++) {
      printf("%f ", input->data[i][j]);
    }
    printf("\n");
  }
}

/*
 * matrix_free
 */
void matrix_free(matrix *m) {
  int i;
  for (i = 0; i < m->rows; i++) {
    free(m->data[i]);
  }
  free(m->data);
  free(m);
  m = NULL;
}

/*
 * matrix_zeros, used as an initialize_function, fills a matrix with zeros.
 */
float matrix_zeros(int i, int j) {
  return 0.0;
}

/*
 * matrix_ones, used as an initialize_function, fills a matrix with ones.
 */
float matrix_ones(int i, int j) {
  return 1.0;
}

/*
 * matrix_random, used as an initialize_function, fills a matrix with random (0, 1).
 */
float matrix_random(int i, int j) {
  return (float)rand()/(float)(RAND_MAX);
}
