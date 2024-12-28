#include "llm/models/gpt2.h"
#include "llm/core/matrix.h"
#include "llm/models/utils.h"
#include <stdlib.h>

GPT2Block create_gpt2_block(GPT2Config config)
{
    GPT2Attention attn = {
        c_attn : create_linear_layer(config.emb_dim, config.emb_dim * 3),
        c_proj : create_linear_layer(config.emb_dim, config.emb_dim)
    };
    GPT2MLP mlp = {
        c_fc : create_linear_layer(config.emb_dim, config.emb_dim * 4),
        c_proj : create_linear_layer(config.emb_dim * 4, config.emb_dim)
    };
    GPT2Block block = {
        attn : attn,
        mlp : mlp,
        ln_1 : create_layer_norm_layer(config.emb_dim),
        ln_2 : create_layer_norm_layer(config.emb_dim)
    };
    return block;
}

GPT2 new_gpt2(GPT2Config config)
{
    GPT2Block *blocks = malloc(sizeof(GPT2Block) * config.num_layers);
    for (int i = 0; i < config.num_layers; i++)
        blocks[i] = create_gpt2_block(config);
    GPT2 gpt2 = {
        config : config,
        wpe : new_matrix(config.vocab_size, config.emb_dim),
        wte : new_matrix(config.ctx_len, config.emb_dim),
        blocks : blocks,
        ln_f : create_layer_norm_layer(config.emb_dim),
        lm_head : new_matrix(config.emb_dim, config.vocab_size)
    };
    return gpt2;
}

void init_gpt2(GPT2 g)
{
    randomize_matrix(g.wte);
    randomize_matrix(g.wpe);
    for (int i = 0; i < g.config.num_layers; i++)
    {
        randomize_matrix(g.blocks[i].attn.c_attn.W);
        randomize_matrix(g.blocks[i].attn.c_attn.b);
        randomize_matrix(g.blocks[i].attn.c_proj.W);
        randomize_matrix(g.blocks[i].attn.c_proj.W);
        randomize_matrix(g.blocks[i].mlp.c_fc.W);
        randomize_matrix(g.blocks[i].mlp.c_fc.b);
        randomize_matrix(g.blocks[i].mlp.c_proj.W);
        randomize_matrix(g.blocks[i].mlp.c_proj.b);
        randomize_matrix(g.blocks[i].ln_1.g);
        randomize_matrix(g.blocks[i].ln_1.b);
        randomize_matrix(g.blocks[i].ln_2.g);
        randomize_matrix(g.blocks[i].ln_2.b);
    }
    randomize_matrix(g.lm_head.W);
    randomize_matrix(g.lm_head.b);
    randomize_matrix(g.ln_f.g);
    randomize_matrix(g.ln_f.b);
}

void load_gpt2(GPT2 g, char *path)
{
}

Matrix generate_gpt2(GPT2 g, int *in)
{
}