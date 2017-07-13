#ifndef __LIST_H_
#define __LIST_H_

typedef struct list_node {
  struct list_node *next;
  struct list_node *previous;
  void *value;
} list_node;

typedef struct list {
  list_node *head;
  list_node *tail;
  int length;
} list;

list *list_create();
list *list_add(list *l, void *value);
list *list_remove(list *l, list_node *n);
void list_free(list *l);

#define list_for_each(l, item) \
  for (item = l->head; item != NULL; item = item->next)

#define list_for_each_reverse(l, item) \
  for (item = l->tail; item != NULL; item = item->previous)

#endif
