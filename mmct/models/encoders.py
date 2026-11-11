import torch
import torch.nn as nn

from typing import Optional

from mmct.utils.utils import sort_batched



class SpaceGroupEncoder(nn.Module):
    """
    Transformer encoder for space group token sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        heads: int,
        fc_dim: int,
        layers: int,
        n_embed: int,
        droprate: float,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embed)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embed,
            nhead=heads,
            dim_feedforward=fc_dim,
            dropout=droprate,
            batch_first=True,
        )
        self.modality_embeddings = nn.Parameter(torch.rand(1, 5, n_embed))
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sg_len)
        x = self.token_embeddings(x)  # (batch_size, sg_len, n_embed)

        x = x + self.modality_embeddings
        # Perform self-attention between the space group tokens.
        x = self.transformer_encoder(x)  # (batch_size, sg_len, n_embed)

        # Apply mean pooling
        x = x.mean(dim=1)  # (batch_size, n_embed)

        return x

class LatticeParametersEncoder(nn.Module):
    """
    Fully connected network to encode 6 lattice parameters.
    """

    def __init__(
        self, inp_len: int, hidden_dim: int, n_embed: int, batch_norm: bool = True
    ):
        super().__init__()
        self.fc1 = nn.Linear(inp_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_embed)
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if batch_norm else None
        # self.missing_embedding = nn.Parameter(torch.rand(n_embed))
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, lp_len)

        # Mask the samples where the LPs are missing.
        # mask = (x == -1).all(dim=1)  # (batch_size,)

        x = self.fc1(x)  # (batch_size, lp_fc_dim)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.act(x)
        x = self.fc2(x)  # (batch_size, n_embed)

        # Set the embedding for samples with missing LPs to the missing
        # embedding. Cast it because mixed precision can make x bf16 while
        # parameters remain fp32.
        # missing_embedding = self.missing_embedding.to(dtype=x.dtype, device=x.device)
        # x = torch.where(mask.unsqueeze(-1), missing_embedding.unsqueeze(0), x)

        return x

class WyckoffEncoder(nn.Module):
    """
    Transformer encoder for element-wyckoff pairs.
    """

    def __init__(
        self,
        pretrained: bool,
        pairwise_attn: bool,
        wk_n_tokens: Optional[int],
        comp_encoder: nn.Module,
        pairwise_heads: int,
        fc_dim: int,
        pairwise_layers: int,
        n_embed: int,
        droprate: float,
        vocab: dict,
        wk_embed: Optional[int] = None,
        wk_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()  # type: ignore
        self.pretrained = pretrained
        self.pairwise_attn = pairwise_attn
        self.vocab = vocab

        if self.pretrained and wk_embed:
            self.wyckoff_embeddings = nn.Embedding.from_pretrained(
                wk_embeddings, freeze=True, padding_idx=vocab["[PAD]"]
            )
            self.proj = nn.Linear(wk_embed, n_embed, bias=False)
        elif wk_n_tokens:
            self.wyckoff_embeddings = nn.Embedding(
                wk_n_tokens + 1, n_embed, padding_idx=vocab["[PAD]"]
            )
        else:
            raise ValueError("Pls pass in either wk_n_tokens or wk_embeddings and wk_embed")

        self.comp_encoder = comp_encoder

        # self.missing_embedding = nn.Parameter(torch.rand(1, 1, n_embed))

        if self.pairwise_attn:
            # self-attention between each wyckoff index and its corresponding element.
            pairwise_encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_embed,
                nhead=pairwise_heads,
                dim_feedforward=fc_dim,
                dropout=droprate,
                batch_first=True,
            )
            self.pairwise_encoder = nn.TransformerEncoder(
                pairwise_encoder_layer, num_layers=pairwise_layers
            )

    def forward(self, x_wke: torch.Tensor, x_wki: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_wke: Tensor of shape (batch_size, wk_len)
            x_wki: Tensor of shape (batch_size, wk_len)
        Returns:
            if self.comp_cross_attn:
                Tensor of shape (batch_size, wk_len, n_embed)
            else:
                Tensor of shape (batch_size, n_embed)
        """

        # Get mask for padding tokens.
        pad_mask = x_wki == self.vocab["[PAD]"]  # (batch_size, comp_len, num_wk_pos)
        non_pad_mask = (
            pad_mask.logical_not().unsqueeze(-1)
        )  # (batch_size, wk_len, 1)

        # Get mask for missing wyckoff positions
        # missing_mask = x_wki == self.vocab["[MISSING]"]  # (batch_size, wk_len)

        # Replace -1s with 0s for safe lookup in the embedding table
        # x_wki[missing_mask] = 0 
        # x_wki (bs, comp_len, num_wk_pos)
        # Embed the wyckoff indices.
        x_wki = self.wyckoff_embeddings(x_wki) # bs, comp_len, num_wk_pos, n_embed

        if self.pretrained:
            # Project wyckoff embeddings to size n_embed
            x_wki = self.proj(x_wki)  # (batch_size, wk_len, n_embed)

        # Replace embeddings at originally missing positions with the learnable
        # missing embedding.
        # x_wki = torch.where(
        #     missing_mask.unsqueeze(-1),
        #     self.missing_embedding.to(dtype=x_wki.dtype, device=x_wki.device),
        #     x_wki,
        # )  # (batch_size, wk_len, n_embed)

        if self.pairwise_attn:
            bs, comp_len, n_embed = x_wki.shape[0], x_wki.shape[1], x_wki.shape[-1]
            x_wki = x_wki.reshape(bs * comp_len, -1, n_embed) # bs * comp_len, num_wk_pos, n_embed
            mask = pad_mask.reshape(bs * comp_len, -1)
            x_wki = self.pairwise_encoder(x_wki, src_key_padding_mask = mask)
            x_wki = x_wki.reshape(bs, comp_len, -1, n_embed)

        # non_pad_mask = non_pad_mask.unsqueeze(-1)
        pad_mask = pad_mask.unsqueeze(-1)
        x_wki = x_wki.masked_fill(pad_mask, 0)
        x_wki = x_wki.sum(dim=2) / (~pad_mask).sum(dim=2) # bs, comp_len, n_embed
        # get padding for wke for padded to comp len
        pad_ele = x_wke == self.vocab["[PAD]"] # bs, comp_len
        x_wki.masked_fill_(pad_ele.unsqueeze(-1), self.vocab["[PAD]"])

        # x_wke[x_wke == self.vocab["[MISSING]"]] = 0
        
        x_wke, x_wki = sort_batched(x_wke, x_wki)

#   x = (x * mask).sum(dim=1) / mask.sum(dim=1)  
        return x_wki

class GeometryEncoder(nn.Module):
    """ """

    def __init__(
        self,
        k,
        geo_apply_weights,
        n_embed
    ):
        super().__init__()
        self.geo_apply_weights = geo_apply_weights
        self.missing_embedding = nn.Parameter(torch.rand(1, 1, k))
        self.fc = nn.Linear(k, n_embed)

    def forward(self, x, geo_t):
        # pdd: (batch_size, comp_len, k + 1)
        # geo_t: (batch_size, comp_len)
        weights = x[:, :, 0]  # (batch_size, max_geo_len)
        x = x[:, :, 1:]  # (batch_size, max_geo_len, k)
        pad_row_mask = x.eq(0).all(dim=-1)
        missing_row_mask = x.eq(-1).all(dim=-1)
        valid_row_mask = ~(pad_row_mask | missing_row_mask)

        # Normalize real PDD rows only; padded rows stay zero and missing rows use
        # a learnable embedding.
        valid = valid_row_mask.unsqueeze(-1).expand_as(x).to(x.dtype)
        count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)  # bs, geo_len, 1

        mean = (x * valid).sum(dim=-1, keepdim=True) / count
        var = (((x - mean) ** 2) * valid).sum(dim=-1, keepdim=True) / count
        eps = 1e-6
        normalized = (x - mean) / torch.sqrt(var + eps)
        x = normalized * valid

        x = torch.where(
            missing_row_mask.unsqueeze(-1),
            self.missing_embedding.to(dtype=x.dtype, device=x.device),
            x,
        )  # (batch_size, max_geo_len, k)
        x = self.fc(x)
        if self.geo_apply_weights:
            x = torch.where(
                valid_row_mask.unsqueeze(-1),
                x * weights.unsqueeze(-1),
                x,
            )  # (batch_size, max_geo_len, k)

        geo_t, x = sort_batched(geo_t, x)

        return x