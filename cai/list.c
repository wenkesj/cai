#include "list.h"
#include <stdlib.h>
#include <stdio.h>

/*
 * list_create
 */
list *list_create() {
  list *l;
  if ((l = malloc(sizeof(*l))) == NULL) {
    perror("Out of memory\n");
    return NULL;
  }
  l->head = l->tail = NULL;
  l->length = 0;
  return l;
}

/*
 * list_add appends a node with the value to the list.
 */
list *list_add(list *l, void *value) {
  list_node *n;
  if ((n = malloc(sizeof(*n))) == NULL) {
    perror("Out of memory\n");
    return NULL;
  }
  n->value = value;
  if (l->length == 0) {
    l->head = l->tail = n;
    n->previous = n->next = NULL;
  } else {
    n->previous = l->tail;
    n->next = NULL;
    l->tail->next = n;
    l->tail = n;
  }
  l->length++;
  return l;
}

/*
 * list_remove
 */
list *list_remove(list *l, list_node *n) {
  if (n != NULL) {
    if (n->previous != NULL) {
      n->previous->next = n->next;
    } else {
      l->head = n->next;
    }
    if (n->next != NULL) {
      n->next->previous = n->previous;
    } else {
      l->tail = n->previous;
    }
    free(n);
    n = NULL;
    l->length--;
  }
}

/*
 * list_free
 */
void list_free(list *l) {
  if (l != NULL) {
    if (l->length > 0) {
      list_node *n = l->head;
      while (n->next != NULL) {
        list_remove(l, n);
        n = n->next;
      }
      free(l);
      l = NULL;
    }
  }
}
