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
 * list_add queues the a node with the value to the list.
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
    // enqueue
    l->tail->next = n;
    n->previous = l->tail;
    l->tail = n;
    n->next = NULL;

    // push
    // n->previous = NULL;
    // n->next = l->head;
    // l->head->previous = n;
    // l->head = n;
  }
  l->length++;
  return l;
}

/*
 * list_remove
 */
list *list_remove(list *l, list_node *n) {
  if (n != NULL) {
    if (n->previous) {
      n->previous->next = n->next;
    } else {
      l->head = n->next;
    }
    if (n->next) {
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
