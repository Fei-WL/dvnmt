# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.transformer_layer import TransformerEncoderLayer
from torch import Tensor


class AutoformerSingleEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.fp16 = args.fp16
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.context_encoder_layer = TransformerEncoderLayer(args)
        self.build_clsr_layer()

        self.cross_attn = self.build_cross_attention(self.embed_dim, args)
        self.cross_attn_layer_norm = LayerNorm(self.embed_dim)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size, bias=True):
        return quant_noise(
            nn.Linear(input_dim, output_dim, bias), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size, bias=True):
        return quant_noise(
            nn.Linear(input_dim, output_dim, bias), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_clsr_layer(self):
        self.G_fc1 = self.build_fc1(self.embed_dim, 128,
                                    self.quant_noise,
                                    self.quant_noise_block_size,
                                    bias=True)
        self.G_fc1_activation_fn = utils.get_activation_fn("relu")
        self.G_fc2 = self.build_fc2(128, 1,
                                    self.quant_noise,
                                    self.quant_noise_block_size,
                                    bias=False)
        self.G_activation_dropout_module = FairseqDropout(
            float(0), module_name=self.__class__.__name__
        )
        self.G_dropout_module = FairseqDropout(
            float(0), module_name=self.__class__.__name__
        )
        self.g_dropout = FairseqDropout(
            float(0), module_name=self.__class__.__name__
        )
        self.g_activation_fn = utils.get_activation_fn("sigmoid")

        self.prev_W = self.build_fc1(self.embed_dim, self.embed_dim,
                                     self.quant_noise,
                                     self.quant_noise_block_size,
                                     bias=False)
        self.post_W = self.build_fc2(self.embed_dim, self.embed_dim,
                                     self.quant_noise,
                                     self.quant_noise_block_size,
                                     bias=False)

    def build_cross_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm",
                          "1": "cross_attn_layer_norm",
                          "2": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]
        self.context_encoder_layer.upgrade_state_dict_named(state_dict, name)

    def do_clsr_gate(self, curr, prev,
                     prev_encoder_padding_mask):
        G_fc1 = self.G_fc1_activation_fn(self.G_fc1(curr))     # output shape:[batch, sent_len, 128]
        G_fc1 = self.G_activation_dropout_module(G_fc1)
        G = self.G_fc2(G_fc1)                               # output shape:[batch, sent_len, 1]
        G = self.G_dropout_module(G)

        if self.training:
            g = self.g_activation_fn(G)
            g = self.g_dropout(g)
            clsr_ctx_padding_mask = prev_encoder_padding_mask
        else:
            g = (G > 0).float() if not self.fp16 else (G > 0).half()
            g_temp = g[:, :, -1].transpose(0, 1)
            clsr_ctx_padding_mask = g_temp * prev_encoder_padding_mask + (1 - g_temp) * prev_encoder_padding_mask

        h_post = self.post_W(prev)
        h_prev = self.prev_W(prev)

        clsr = g * h_post + (1 - g) * h_prev

        return clsr, clsr_ctx_padding_mask

    def forward(self, curr_x, prev_x,
                curr_encoder_padding_mask,
                prev_encoder_padding_mask,
                curr_attn_mask: Optional[Tensor] = None,
                prev_attn_mask: Optional[Tensor] = None,):
        if curr_attn_mask is not None:
            curr_attn_mask = curr_attn_mask.masked_fill(curr_attn_mask.to(torch.bool), -1e8)
        if prev_attn_mask is not None:
            prev_attn_mask = prev_attn_mask.masked_fill(prev_attn_mask.to(torch.bool), -1e8)

        """
        Self-Attention
        """
        residual = curr_x
        if self.normalize_before:
            curr_x = self.self_attn_layer_norm(curr_x)
        curr_x, _ = self.self_attn(
            query=curr_x,
            key=curr_x,
            value=curr_x,
            key_padding_mask=curr_encoder_padding_mask,
            attn_mask=curr_attn_mask,
        )
        curr_x = self.dropout_module(curr_x)
        curr_x = self.residual_connection(curr_x, residual)
        if not self.normalize_before:
            curr_x = self.self_attn_layer_norm(curr_x)

        """
        Use clsr to make context & Do cross-attention
        """
        prev_x = self.context_encoder_layer(prev_x, prev_encoder_padding_mask, prev_attn_mask)
        clsr_ctx, clsr_ctx_padding_mask = self.do_clsr_gate(curr_x, prev_x,
                                                            prev_encoder_padding_mask)
        residual = curr_x
        if self.normalize_before:
            curr_x = self.cross_attn_layer_norm(curr_x)
        curr_x, _ = self.cross_attn(query=curr_x,
                                    key=clsr_ctx,
                                    value=clsr_ctx,
                                    key_padding_mask=clsr_ctx_padding_mask,
                                    static_kv=True)
        curr_x = self.dropout_module(curr_x)
        curr_x = self.residual_connection(curr_x, residual)
        if not self.normalize_before:
            curr_x = self.self_attn_layer_norm(curr_x)

        """
        Feed-Forward
        """
        residual = curr_x
        if self.normalize_before:
            curr_x = self.final_layer_norm(curr_x)
        curr_x = self.activation_fn(self.fc1(curr_x))
        curr_x = self.activation_dropout_module(curr_x)
        curr_x = self.fc2(curr_x)
        curr_x = self.dropout_module(curr_x)
        curr_x = self.residual_connection(curr_x, residual)
        if not self.normalize_before:
            curr_x = self.final_layer_norm(curr_x)

        return curr_x, prev_x