from __future__ import annotations

import torch
from torch import nn

from fiesl.data import EvidenceBatch
from fiesl.ontology import PRIMITIVE_ORDER, SOURCE_SLOT, unit_membership


PRIMITIVE_NAMES = PRIMITIVE_ORDER
TEXT_PRIMITIVES = (0, 2, 7)


class SetReadout(nn.Module):
    def __init__(self, hidden_dim: int, inner_dim: int, dropout: float) -> None:
        super().__init__()
        self.element = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        transformed = self.element(values)
        weights = mask.unsqueeze(-1).to(transformed.dtype)
        pooled = (transformed * weights).sum(dim=1)
        pooled = pooled / weights.sum(dim=1).clamp_min(1.0)
        return self.output(pooled)


class PrimitiveAdapter(nn.Module):
    def __init__(self, input_dims: list[int]) -> None:
        super().__init__()
        if len(input_dims) != 9:
            raise ValueError("FIESL requires nine evidence slots")
        encoder_dim = int(input_dims[0]) - 15
        expected = [encoder_dim + 15, encoder_dim + 5, 1, 4, 4, 4, encoder_dim, 9, 15]
        if [int(value) for value in input_dims] != expected:
            raise ValueError("input dimensions do not match the FIESL evidence contract")
        self.input_dims = expected
        self.encoder_dim = encoder_dim
        self.width = encoder_dim + 10

    def pad(self, values: torch.Tensor) -> torch.Tensor:
        result = values.new_zeros((values.shape[0], self.width))
        result[:, : values.shape[1]] = values
        return result

    def forward(self, batch: EvidenceBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        typed = batch.typed_inputs
        dim = self.encoder_dim
        values = [
            torch.cat((typed[:, 0, :dim], typed[:, 0, dim : dim + 10]), dim=1),
            typed[:, 0, dim + 10 : dim + 15],
            torch.cat((typed[:, 1, :dim], typed[:, 1, dim + 3 : dim + 5]), dim=1),
            typed[:, 1, dim : dim + 3],
            typed[:, 3, :4],
            typed[:, 4, :4],
            typed[:, 5, :4],
            typed[:, 6, :dim],
            typed[:, 7, :9],
            typed[:, 8, :15],
        ]
        projected = torch.stack([self.pad(value) for value in values], dim=1)
        source = torch.tensor(SOURCE_SLOT, dtype=torch.int64, device=typed.device)
        mask = batch.availability_mask.index_select(1, source)
        quality = batch.quality_features.index_select(1, source)
        projected = projected * mask.unsqueeze(-1).to(projected.dtype)
        quality = quality * mask.unsqueeze(-1).to(quality.dtype)
        return projected, mask, quality


class SharedPrimitiveEncoder(nn.Module):
    def __init__(
        self,
        width: int,
        quality_dim: int,
        hidden_dim: int,
        inner_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        def projection() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, inner_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, hidden_dim),
            )

        self.text_projection = projection()
        self.numeric_projection = projection()
        self.quality_projection = nn.Sequential(
            nn.LayerNorm(quality_dim),
            nn.Linear(quality_dim, hidden_dim),
        )
        self.type_embedding = nn.Embedding(len(PRIMITIVE_NAMES), hidden_dim)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        quality: torch.Tensor,
    ) -> torch.Tensor:
        output = values.new_zeros((*values.shape[:2], self.hidden_dim))
        text_set = set(TEXT_PRIMITIVES)
        for index in range(len(PRIMITIVE_NAMES)):
            projection = self.text_projection if index in text_set else self.numeric_projection
            output[:, index] = projection(values[:, index])
        output = output + self.quality_projection(quality)
        return output * mask.unsqueeze(-1).to(output.dtype)


class FIESL(nn.Module):
    def __init__(
        self,
        input_dims: list[int],
        quality_dim: int,
        hidden_dim: int = 128,
        encoder_inner_dim: int = 256,
        pair_hidden_dim: int = 128,
        aggregation_hidden_dim: int = 256,
        encoder_dropout: float = 0.1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.adapter = PrimitiveAdapter(input_dims)
        self.encoder = SharedPrimitiveEncoder(
            self.adapter.width,
            quality_dim,
            hidden_dim,
            encoder_inner_dim,
            encoder_dropout,
        )
        membership = unit_membership()
        self.register_buffer("unit_membership", membership)
        pair_left, pair_right = torch.triu_indices(9, 9, offset=1)
        incidence = torch.zeros((9, pair_left.numel()), dtype=torch.bool)
        columns = torch.arange(pair_left.numel())
        incidence[pair_left, columns] = True
        incidence[pair_right, columns] = True
        self.register_buffer("pair_left", pair_left)
        self.register_buffer("pair_right", pair_right)
        self.register_buffer("incidence", incidence)
        pair_width = 3 * hidden_dim
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(pair_width),
            nn.Linear(pair_width, pair_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden_dim, hidden_dim),
        )
        self.pair_gate = nn.Sequential(
            nn.Linear(pair_width, pair_hidden_dim),
            nn.GELU(),
            nn.Linear(pair_hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.unit_fusion = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.semantic_readout = SetReadout(hidden_dim, aggregation_hidden_dim, dropout)
        self.structural_readout = SetReadout(hidden_dim, aggregation_hidden_dim, dropout)
        self.structural_projection = nn.Linear(hidden_dim, hidden_dim)
        self.structural_scale = nn.Parameter(torch.tensor(0.1))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def encode_units(self, batch: EvidenceBatch) -> tuple[torch.Tensor, torch.Tensor]:
        values, primitive_mask, quality = self.adapter(batch)
        primitives = self.encoder(values, primitive_mask, quality)
        weights = primitive_mask.unsqueeze(1).to(primitives.dtype) * self.unit_membership.unsqueeze(0)
        counts = weights.sum(dim=2)
        units = torch.einsum("bph,bup->buh", primitives, weights)
        units = units / counts.clamp_min(1).unsqueeze(-1)
        normalized_membership = self.unit_membership / self.unit_membership.sum(dim=1, keepdim=True)
        unit_types = normalized_membership @ self.encoder.type_embedding.weight
        unit_mask = counts > 0
        units = (units + unit_types.unsqueeze(0)) * unit_mask.unsqueeze(-1).to(units.dtype)
        return units, unit_mask

    def forward(self, batch: EvidenceBatch) -> torch.Tensor:
        units, unit_mask = self.encode_units(batch)
        left = units.index_select(1, self.pair_left)
        right = units.index_select(1, self.pair_right)
        pair_features = torch.cat((torch.abs(left - right), left * right, left + right), dim=2)
        pair_mask = unit_mask.index_select(1, self.pair_left) & unit_mask.index_select(1, self.pair_right)
        pair_values = self.pair_encoder(pair_features)
        gated = pair_values * self.pair_gate(pair_features) * pair_mask.unsqueeze(-1).to(pair_values.dtype)
        incident_mask = self.incidence.unsqueeze(0) & pair_mask.unsqueeze(1)
        profiles = torch.einsum("bph,bup->buh", gated, incident_mask.to(gated.dtype))
        profiles = profiles / incident_mask.sum(dim=2).clamp_min(1).unsqueeze(-1).to(profiles.dtype)
        enhanced = self.unit_fusion(torch.cat((units, profiles), dim=2))
        enhanced = enhanced * unit_mask.unsqueeze(-1).to(enhanced.dtype)
        semantic = self.semantic_readout(units, unit_mask)
        structural = self.structural_readout(enhanced, unit_mask)
        account = self.output_norm(semantic + self.structural_scale * self.structural_projection(structural))
        return self.classifier(account)
