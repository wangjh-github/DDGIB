import torch.nn as nn
import torch
import math
import random


class Graph_sampler(nn.Module):

    def forward(self, graphs, p, min_len, max_len):

        length = len(graphs)
        size = 0

        pos = random.randint(0, length - min_len)
        for i in range(min_len, max_len + 1):
            size = i
            if pos + size >= length:
                break
            temp_p = random.random()
            if temp_p > p:
                break
        return graphs[pos:pos + size], pos, pos + size
