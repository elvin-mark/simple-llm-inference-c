#include <stdio.h>
#include <stdlib.h>
#include "llm/core/utils.h"

void assert(int cond, char *message)
{
    if (!cond)
    {
        printf("Error: %s\n", message);
        exit(-1);
    }
}