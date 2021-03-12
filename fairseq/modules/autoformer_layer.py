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


class AutoformerEncoderLayer(nn.Module):
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
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
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

    def do_clsr_gate(self, curr, prev, post,
                     prev_encoder_padding_mask,
                     post_encoder_padding_mask):
        G_fc1 = self.G_fc1_activation_fn(self.G_fc1(curr))     # output shape:[batch, sent_len, 128]
        G_fc1 = self.G_activation_dropout_module(G_fc1)
        G = self.G_fc2(G_fc1)                               # output shape:[batch, sent_len, 1]
        G = self.G_dropout_module(G)

        if self.training:
            g = self.g_activation_fn(G)
            g = self.g_dropout(g)
            clsr_ctx_padding_mask = prev_encoder_padding_mask & post_encoder_padding_mask
        else:
            if self.fp16:
                g = (G > 0).half()
            else:
                g = (G > 0).float()
            g_temp = g[:, :, -1].transpose(0, 1)
            clsr_ctx_padding_mask = g_temp * prev_encoder_padding_mask + (1 - g_temp) * post_encoder_padding_mask

        h_prev = self.prev_W(prev)
        h_post = self.post_W(post)

        clsr = g * h_prev + (1 - g) * h_post
        return clsr, clsr_ctx_padding_mask

    def forward(self, curr_x, prev_x, post_x,
                curr_encoder_padding_mask,
                prev_encoder_padding_mask,
                post_encoder_padding_mask,
                curr_attn_mask: Optional[Tensor] = None,
                prev_attn_mask: Optional[Tensor] = None,
                post_attn_mask: Optional[Tensor] = None):
        if curr_attn_mask is not None:
            curr_attn_mask = curr_attn_mask.masked_fill(curr_attn_mask.to(torch.bool), -1e8)
        if prev_attn_mask is not None:
            prev_attn_mask = prev_attn_mask.masked_fill(prev_attn_mask.to(torch.bool), -1e8)
        if post_attn_mask is not None:
            post_attn_mask = post_attn_mask.masked_fill(post_attn_mask.to(torch.bool), -1e8)

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
        post_x = self.context_encoder_layer(post_x, post_encoder_padding_mask, post_attn_mask)
        clsr_ctx, clsr_ctx_padding_mask = self.do_clsr_gate(curr_x, prev_x, post_x,
                                                            prev_encoder_padding_mask,
                                                            post_encoder_padding_mask)
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

        return curr_x, prev_x, post_x

class AutoformerDecoderLayer(nn.Module):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.build_clsr_layer()

        self.clsr_attn = self.build_encoder_attention(self.embed_dim, args)
        self.clsr_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.cross_attn = self.build_encoder_attention(self.embed_dim, args)
        self.cross_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size, bias=True):
        return quant_noise(nn.Linear(input_dim, output_dim, bias), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size, bias=True):
        return quant_noise(nn.Linear(input_dim, output_dim, bias), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
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

    def do_clsr_gate(self, curr, prev, post, tgt,
                     tgt_decoder_padding_mask,
                     prev_encoder_padding_mask,
                     post_encoder_padding_mask):
        curr, _ = self.clsr_attn(query=curr,
                              key=tgt,
                              value=tgt,
                              key_padding_mask=tgt_decoder_padding_mask,
                              static_kv=True)
        curr = self.dropout_module(curr)
        curr = self.clsr_attn_layer_norm(curr)

        G_fc1 = self.G_fc1_activation_fn(self.G_fc1(curr))     # output shape:[batch, sent_len, 128]
        G_fc1 = self.G_activation_dropout_module(G_fc1)
        G = self.G_fc2(G_fc1)                               # output shape:[batch, sent_len, 1]
        G = self.G_dropout_module(G)

        if self.training:
            g = self.g_activation_fn(G)
            g = self.g_dropout(g)
            clsr_ctx_padding_mask = prev_encoder_padding_mask & post_encoder_padding_mask
        else:
            g = (G > 0).float()
            g_temp = g[:, :, -1].transpose(0, 1)
            clsr_ctx_padding_mask = g_temp * prev_encoder_padding_mask + (1 - g_temp) * post_encoder_padding_mask

        h_prev = self.prev_W(prev)
        h_post = self.post_W(post)

        clsr = g * h_prev + (1 - g) * h_post

        return clsr, clsr_ctx_padding_mask

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        prev_out: Optional[torch.Tensor] = None,
        prev_out_padding_mask: Optional[torch.Tensor] = None,
        post_out: Optional[torch.Tensor] = None,
        post_out_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        clsr, clsr_padding_mask = self.do_clsr_gate(curr=encoder_out,
                                                    prev=prev_out,
                                                    post=post_out,
                                                    tgt=x,
                                                    prev_encoder_padding_mask=prev_out_padding_mask,
                                                    post_encoder_padding_mask=post_out_padding_mask,
                                                    tgt_decoder_padding_mask=self_attn_padding_mask)
        residual = x
        if self.normalize_before:
            x = self.cross_attn_layer_norm(x)
        x, _ = self.cross_attn(query=x,
                               key=clsr,
                               value=clsr,
                               key_padding_mask=clsr_padding_mask,
                               static_kv=True)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.cross_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn