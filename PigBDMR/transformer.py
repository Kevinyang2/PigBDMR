# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
import numpy as np
from .attention import MultiheadAttention
from .crossattention import MultiheadAttention as cateattention

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def gen_sineembed_for_position(pos_tensor, d_model):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(d_model//2, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (d_model//2))
    center_embed = pos_tensor[:, :, 0] * scale
    pos_x = center_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

    span_embed = pos_tensor[:, :, 1] * scale
    pos_w = span_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

    pos = torch.cat((pos_x, pos_w), dim=2)
    return pos

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 args=None
                 ):
        super().__init__()
        if args is None:
            raise ValueError("Transformer args are required")
        self.args = args
        self.fusion_mode = getattr(args, "fusion_mode", "full")

        # T2V Cross-Attention (used in cross-attention modes)
        needs_t2v = self.fusion_mode in ("vanilla_cross_attn", "vanilla_cross_attn_encoder", "cross_attn", "full")
        if needs_t2v:
            t2v_encoder_layer = T2V_TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before, self.args.num_dummies,
                                                    getattr(self.args, "fusion_variant", "baseline"),
                                                    getattr(self.args, "use_null_gate", False))
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.t2v_encoder = TransformerCATEEncoder(t2v_encoder_layer, args.t2v_layers, encoder_norm)
        else:
            self.t2v_encoder = None

        # Fusion Encoder (used in all modes except cross_attn)
        needs_encoder = self.fusion_mode != "cross_attn"
        if needs_encoder:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        else:
            self.encoder = None

        self._reset_parameters()
        if self.t2v_encoder is not None:
            for layer in self.t2v_encoder.layers:
                if isinstance(layer, T2V_TransformerEncoderLayer):
                    layer.reset_fusion_parameters()

        self.d_model = d_model
        self.nhead = nhead

        # Feature-dim concat projection (2*d → d)
        if self.fusion_mode == "concat":
            self.concat_proj = nn.Linear(d_model * 2, d_model)
        else:
            self.concat_proj = None

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
                nn.init.trunc_normal_(p, std=.02)

    def forward(self, src, mask, pos_embed, video_length=None, saliency_proj1=None, saliency_proj2=None, query_emb=None):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            pos_embed: (batch_size, L, d) the same as src
            video_length: number of video tokens
        Returns:
        """
        assert saliency_proj1 is not None
        assert saliency_proj2 is not None
        assert video_length is not None
        bs, l, d = src.shape
        src = src.permute(1, 0, 2)  # (L, batch_size, d)
        pos_embed = pos_embed.permute(1, 0, 2)   # (L, batch_size, d)
        attn_weights = None

        if self.fusion_mode == "add":
            # Element-wise addition: mean-pool valid text tokens, add to each video token
            text_start = video_length + self.args.num_dummies
            text_src = src[text_start:]
            text_mask = mask[:, text_start:]  # (B, L_txt), True=masked
            valid = (~text_mask).transpose(0, 1).unsqueeze(-1).to(text_src.dtype)  # (L_txt, B, 1)
            denom = valid.sum(dim=0).clamp(min=1.0)  # (B, 1)
            text_mean = (text_src * valid).sum(dim=0) / denom  # (B, d)
            vid_fuse = src[:video_length] + text_mean.unsqueeze(0)  # broadcast add
            vid_mask = mask[:, :video_length]
            vid_pos = pos_embed[:video_length]

        elif self.fusion_mode == "add_with_dummy":
            # Legacy add baseline: mean-pool dummy tokens together with real text tokens.
            text_src = src[video_length:]
            text_mask = mask[:, video_length:]  # (B, L_txt), True=masked
            valid = (~text_mask).transpose(0, 1).unsqueeze(-1).to(text_src.dtype)  # (L_txt, B, 1)
            denom = valid.sum(dim=0).clamp(min=1.0)  # (B, 1)
            text_mean = (text_src * valid).sum(dim=0) / denom  # (B, d)
            vid_fuse = src[:video_length] + text_mean.unsqueeze(0)  # broadcast add
            vid_mask = mask[:, :video_length]
            vid_pos = pos_embed[:video_length]

        elif self.fusion_mode == "concat":
            assert self.concat_proj is not None
            # Feature-dimension concatenation: [video_token, text_mean] per token, then project
            text_src = src[video_length:]
            text_mask = mask[:, video_length:]
            valid = (~text_mask).transpose(0, 1).unsqueeze(-1).to(text_src.dtype)
            denom = valid.sum(dim=0).clamp(min=1.0)
            text_mean = (text_src * valid).sum(dim=0) / denom  # (B, d)
            text_broadcast = text_mean.unsqueeze(0).expand(video_length, -1, -1)  # (V, B, d)
            vid_only = src[:video_length]  # (V, B, d)
            vid_fuse = torch.cat([vid_only, text_broadcast], dim=-1)  # (V, B, 2d)
            # Project back to model dim with a simple linear
            bsz_inner = vid_fuse.size(1)
            vid_fuse = vid_fuse.view(video_length * bsz_inner, 2 * d)
            vid_fuse = self.concat_proj(vid_fuse).view(video_length, bsz_inner, d)
            vid_mask = mask[:, :video_length]
            vid_pos = pos_embed[:video_length]

        elif self.fusion_mode == "self_attn":
            # Sequence concatenation: video+text through self-attention encoder
            vid_fuse = src  # full sequence (video + text)
            vid_mask = mask  # full mask
            vid_pos = pos_embed

        elif self.fusion_mode == "vanilla_cross_attn":
            assert self.t2v_encoder is not None
            # Standard cross-attention (includes dummy tokens in attention output)
            t2v_src, attn_weights = self.t2v_encoder(src, src_key_padding_mask=mask, pos=pos_embed, video_length=video_length, dummy=False, query_emb=query_emb)
            vid_fuse = t2v_src[:video_length]
            vid_mask = mask[:, :video_length]
            vid_pos = pos_embed[:video_length]

        elif self.fusion_mode == "vanilla_cross_attn_encoder":
            assert self.t2v_encoder is not None
            # Standard cross-attention without dummy-gating, skip dummy tokens + encoder
            t2v_src, attn_weights = self.t2v_encoder(src, src_key_padding_mask=mask, pos=pos_embed, video_length=video_length, dummy=True, query_emb=query_emb)
            vid_fuse = t2v_src[:video_length]
            vid_mask = mask[:, :video_length]
            vid_pos = pos_embed[:video_length]

        elif self.fusion_mode == "cross_attn":
            assert self.t2v_encoder is not None
            # Adaptive cross-attention with dummy gating, no post encoder
            t2v_src, attn_weights = self.t2v_encoder(src, src_key_padding_mask=mask, pos=pos_embed, video_length=video_length, query_emb=query_emb)
            vid_fuse = t2v_src[:video_length]
            vid_mask = mask[:, :video_length]
            vid_pos = pos_embed[:video_length]

        elif self.fusion_mode == "full":
            assert self.t2v_encoder is not None
            # Full T2V: cross-attention + fusion encoder
            t2v_src, attn_weights = self.t2v_encoder(src, src_key_padding_mask=mask, pos=pos_embed, video_length=video_length, query_emb=query_emb)
            vid_fuse = t2v_src[:video_length]
            vid_mask = mask[:, :video_length]
            vid_pos = pos_embed[:video_length]
        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")

        if self.fusion_mode not in ("vanilla_cross_attn", "cross_attn"):
            assert self.encoder is not None
            vid_fuse = self.encoder(vid_fuse, src_key_padding_mask=vid_mask, pos=vid_pos)  # (L, batch_size, d)

        # For self_attn mode, extract only video portion for downstream
        if self.fusion_mode == "self_attn":
            vid_fuse = vid_fuse[:video_length]
            vid_mask = vid_mask[:, :video_length]
            vid_pos = vid_pos[:video_length]

        vid_mem = vid_fuse.transpose(0, 1)
        memory_global = vid_mem.mean(1)
        proj1_result = saliency_proj1(vid_mem)
        proj2_result = saliency_proj2(memory_global)
        proj2_result = proj2_result.unsqueeze(1)

        intermediate_result = proj1_result * proj2_result
        saliency_scores = torch.sum(intermediate_result, dim=-1) / np.sqrt(d)

        return vid_fuse, vid_mask, vid_pos, attn_weights, saliency_scores

class TransformerCATEEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                dummy=True,
                **kwargs):
        output = src

        intermediate = []
        attn_weights = None
        for i, layer in enumerate(self.layers):
            output, attn_weight = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, dummy=dummy, **kwargs)
            if attn_weights is None:
                attn_weights = attn_weight
            else:
                attn_weights = attn_weights + attn_weight
            if self.return_intermediate:
                intermediate.append(output)
        attn_weights /= self.num_layers

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output, attn_weights

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        output = src

        intermediate = []

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, **kwargs)
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayerThin(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.linear(src2)
        src = src + self.dropout(src2)
        src = self.norm(src)
        # src = src + self.dropout1(src2)
        # src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """not used"""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class T2V_TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, num_dummies=3, fusion_variant="baseline", use_null_gate=False):
        super().__init__()
        if fusion_variant not in ("baseline", "gated_xattn", "film", "gated_xattn_film"):
            raise ValueError(f"Unsupported fusion_variant: {fusion_variant}")
        self.fusion_variant = fusion_variant
        self.use_null_gate = use_null_gate
        self._last_null_gate = None
        self.self_attn = cateattention(d_model, nhead, dropout=dropout, num_dummies=num_dummies)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = DropPath(dropout)
        self.dropout2 = DropPath(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead
        if self.fusion_variant in ("gated_xattn", "gated_xattn_film"):
            self.xattn_gate = nn.Linear(d_model * 2, d_model)
        if self.fusion_variant in ("film", "gated_xattn_film"):
            self.film = nn.Linear(d_model, d_model * 2)
        if self.use_null_gate:
            self.null_gate = nn.Linear(d_model * 2, 1)
        self.reset_fusion_parameters()

    def reset_fusion_parameters(self):
        if self.fusion_variant in ("gated_xattn", "gated_xattn_film"):
            nn.init.zeros_(self.xattn_gate.weight)
            nn.init.constant_(self.xattn_gate.bias, 4.0)
        if self.fusion_variant in ("film", "gated_xattn_film"):
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        if self.use_null_gate:
            nn.init.zeros_(self.null_gate.weight)
            nn.init.constant_(self.null_gate.bias, 3.0)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     video_length=None, dummy=True, query_emb=None):
        assert video_length is not None
        assert src_key_padding_mask is not None
        pos_src = self.with_pos_embed(src, pos)
        q, k, v = pos_src[:video_length], pos_src[video_length:], src[video_length:]


        qmask, kmask = src_key_padding_mask[:, :video_length].unsqueeze(2), src_key_padding_mask[:,
                                                                                 video_length:].unsqueeze(1)
        attn_mask = torch.matmul(qmask.float(), kmask.float()).bool().repeat(self.nhead, 1, 1)

        # - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
        #   If a FloatTensor is provided, it will be directly added to the value.
        #   If a BoolTensor is provided, the positions with the
        #   value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        # - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        #   3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
        #   S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
        #   positions. If a BoolTensor is provided, positions with ``True``
        #   are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
        #   is provided, it will be added to the attention weight.
        # print(q.shape, k.shape, v.shape, attn_mask.shape, src_key_padding_mask[:, video_length + 1:].shape)
        src2, attn_weights = self.self_attn(q, k, v, attn_mask=attn_mask,
                                            key_padding_mask=src_key_padding_mask[:, video_length:], dummy=dummy)

        if self.fusion_variant in ("gated_xattn", "gated_xattn_film"):
            gate = torch.sigmoid(self.xattn_gate(torch.cat([src[:video_length], src2], dim=-1)))
            src2 = gate * src2

        if self.use_null_gate:
            if query_emb is None:
                raise RuntimeError("query_emb is required when --use_null_gate is enabled")
            query_context = query_emb.transpose(0, 1).expand(video_length, -1, -1)
            null_gate = torch.sigmoid(self.null_gate(torch.cat([src[:video_length], query_context], dim=-1)))
            self._last_null_gate = null_gate
            src2 = null_gate * src2

        src2 = src[:video_length] + self.dropout1(src2)

        if self.fusion_variant in ("film", "gated_xattn_film"):
            text_src = src[video_length:]
            text_mask = src_key_padding_mask[:, video_length:]
            valid = (~text_mask).transpose(0, 1).unsqueeze(-1).to(text_src.dtype)
            denom = valid.sum(dim=0).clamp(min=1.0)
            query_context = (text_src * valid).sum(dim=0) / denom
            gamma, beta = self.film(query_context).chunk(2, dim=-1)
            src2 = src2 * (1 + gamma.unsqueeze(0)) + beta.unsqueeze(0)

        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)

        src = torch.cat([src2, src[video_length:]])
        return src, attn_weights

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None, dummy=True):
        pass


    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, dummy=True,
                **kwargs):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, dummy=dummy)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, dummy=dummy, **kwargs)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = DropPath(dropout)
        self.dropout2 = DropPath(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        pass

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        normalize_before=args.pre_norm,
        activation='prelu',
        args=args
    )

def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    x = x.div(keep_prob) * mask

    return x

class DropPath(nn.Module):
    """
    Drop paths per sample (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()

        self.drop_prob = drop_prob

    def forward(self, x):
        x = x.permute(1, 0, 2)
        res = drop_path(x, self.drop_prob, self.training)
        return res.permute(1, 0, 2)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
