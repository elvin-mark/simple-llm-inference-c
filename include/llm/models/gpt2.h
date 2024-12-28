#ifndef GPT2_H
#define GPT2_H

#include "llm/core/matrix.h"
#include "llm/models/utils.h"

typedef struct GPT2Config
{
    int vocab_size;
    int ctx_len;
    int emb_dim;
    int num_heads;
    int num_layers;

} GPT2Config;

typedef struct GPT2Attention
{
    LinearLayer c_attn;
    LinearLayer c_proj;
} GPT2Attention;

typedef struct GPT2MLP
{
    LinearLayer c_fc;
    LinearLayer c_proj;
} GPT2MLP;

typedef struct GPT2Block
{
    GPT2Attention attn;
    GPT2MLP mlp;
    LayerNorm ln_1;
    LayerNorm ln_2;
} GPT2Block;

typedef struct GPT2
{
    GPT2Config config;
    Matrix wte; // Wort Token Embedding
    Matrix wpe; // Word positional Embedding
    GPT2Block *blocks;
    LayerNorm ln_f;
    LinearLayer lm_head;

} GPT2;

GPT2 new_gpt2(GPT2Config config);
void init_gpt2(GPT2 g);
void load_gpt2(GPT2 g, char *path);
Matrix generate_gpt2(GPT2 g, int *in);

#endif