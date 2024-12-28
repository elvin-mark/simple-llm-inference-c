#ifndef BERT_H
#define BERT_H

#include "llm/core/matrix.h"

typedef struct BERTConfig
{

} BERTConfig;

typedef struct BERT
{

} BERT;

BERT new_bert(BERTConfig config);
void init_bert(BERT g);
void load_bert(BERT g, char *path);
Matrix generate_bert(BERT g, int *in);

#endif