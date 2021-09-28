import unittest
from torch import nn
import torch
from einops import repeat

from MultiHeadAttention import MultiHeadAttention


class test_MultiHeadAttention(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.query_dim = 16
        self.context_dim = 32
        self.batch_size = 8
        self.latent_dim = 6
        x = nn.Parameter(torch.randn((self.latent_dim, self.query_dim)))
        context = nn.Parameter(torch.randn((self.latent_dim, self.context_dim)))

        self.x = repeat(x, 'n d -> b n d', b=self.batch_size)
        self.context = repeat(context, 'n d -> b n d', b=self.batch_size)

    def test_selfAttention(self):
        multiHeadAttention = MultiHeadAttention(self.query_dim)
        out = multiHeadAttention.forward(self.x)
        self.assertEqual(self.x.shape, out.shape)

    def test_crossAttention(self):
        multiHeadAttention = MultiHeadAttention(self.query_dim, context_dim=self.context_dim)
        out = multiHeadAttention.forward(self.x, context=self.context)
        self.assertEqual(self.x.shape, out.shape)
