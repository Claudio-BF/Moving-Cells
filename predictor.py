from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Top-level dimensions and hyperparameters
# ============================================================

# Number of spatial position coordinates, e.g. 2 for x,y or 3 for x,y,z
position_dim = 2

# Number of non-position, non-target features per cell
additional_dim = 4

# Number of target dimensions, e.g. 2 for 2D velocity
target_dim = 2

# Full input dimension of each cell vector:
# [position | additional features | target]
total_dim = position_dim + additional_dim + target_dim

# Deep Sets aggregate dimension
aggregate_dim = 128

# Hidden dimensions for phi, the per-context-cell Deep Sets encoder
phi_hidden_dims = (128, 128)

# Hidden dimensions for rho, the target predictor
rho_hidden_dims = (128, 128)

# "mean" is often numerically more stable than "sum" when cell counts vary.
aggregation: Literal["sum", "mean"] = "mean"

# Convention for displacement positions inside the aggregate network.
# For picked/query cell i and context/other cell j:
#     "other_minus_query" means displacement = position_j - position_i.
#     "query_minus_other" means displacement = position_i - position_j.
displacement_convention: Literal["other_minus_query", "query_minus_other"] = (
    "other_minus_query"
)

# This preserves the original instruction that rho sees the picked cell's position.
# If you want the whole learned rule to be strictly translation-invariant, set this to False.
use_query_position_in_rho = True

dropout = 0.0

learning_rate = 1e-3
weight_decay = 1e-5
batch_size = 32
epochs = 30
max_grad_norm = 1.0


# ============================================================
# Config
# ============================================================


@dataclass(frozen=True)
class DeepSetsConfig:
    position_dim: int
    additional_dim: int
    target_dim: int
    aggregate_dim: int = 128
    phi_hidden_dims: Tuple[int, ...] = (128, 128)
    rho_hidden_dims: Tuple[int, ...] = (128, 128)
    aggregation: Literal["sum", "mean"] = "mean"
    displacement_convention: Literal[
        "other_minus_query",
        "query_minus_other",
    ] = "other_minus_query"
    use_query_position_in_rho: bool = True
    dropout: float = 0.0

    @property
    def total_dim(self) -> int:
        return self.position_dim + self.additional_dim + self.target_dim

    @property
    def target_start(self) -> int:
        return self.position_dim + self.additional_dim

    @property
    def context_dim(self) -> int:
        # The aggregate network still sees a vector of the same length as a cell vector:
        # [relative position | additional features | target].
        return self.total_dim

    @property
    def rho_input_dim(self) -> int:
        if self.use_query_position_in_rho:
            return self.position_dim + self.aggregate_dim
        return self.aggregate_dim


# ============================================================
# Utility modules
# ============================================================


def make_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    pred:   [batch, max_cells, target_dim]
    target: [batch, max_cells, target_dim]
    mask:   [batch, max_cells], True for real cells, False for padding
    """
    per_cell_loss = (pred - target).pow(2).mean(dim=-1)
    mask_float = mask.to(per_cell_loss.dtype)

    return (per_cell_loss * mask_float).sum() / mask_float.sum().clamp_min(1.0)


# ============================================================
# Relative-position leave-one-out Deep Sets model
# ============================================================


class LeaveOneOutDeepSets(nn.Module):
    """
    For each picked/query cell i:

        1. Replace each other cell j's absolute position by a displacement
           relative to the picked cell i.

               context_for_i_j = [position_j - position_i | additional_j | target_j]

           The sign can be changed with config.displacement_convention.

        2. Encode every other cell j != i using phi.
        3. Aggregate those encodings.
        4. Concatenate:
              [position of cell i, aggregate of relative-position other cells]
           unless config.use_query_position_in_rho is False, in which case rho sees
           only the aggregate.
        5. Predict the target vector of cell i using rho.

    The implementation computes all picked cells in parallel. Because each picked
    cell has a different relative coordinate system, phi is applied to a pairwise
    tensor of shape [batch, query_cells, context_cells, total_dim].
    """

    def __init__(self, config: DeepSetsConfig):
        super().__init__()
        self.config = config

        self.phi = make_mlp(
            input_dim=config.context_dim,
            hidden_dims=config.phi_hidden_dims,
            output_dim=config.aggregate_dim,
            dropout=config.dropout,
        )

        self.rho = make_mlp(
            input_dim=config.rho_input_dim,
            hidden_dims=config.rho_hidden_dims,
            output_dim=config.target_dim,
            dropout=config.dropout,
        )

    def positions_from_cells(self, cells: torch.Tensor) -> torch.Tensor:
        return cells[..., : self.config.position_dim]

    def targets_from_cells(self, cells: torch.Tensor) -> torch.Tensor:
        start = self.config.target_start
        end = start + self.config.target_dim
        return cells[..., start:end]

    def relative_context_from_queries(
        self,
        query_positions: torch.Tensor,
        context_cells: torch.Tensor,
    ) -> torch.Tensor:
        """
        Builds the tensor seen by phi.

        query_positions: [batch, num_queries, position_dim]
        context_cells:   [batch, num_context_cells, total_dim]

        Returns:
            relative_context: [batch, num_queries, num_context_cells, total_dim]

        The last dimension is:
            [relative/displacement position | context additional features | context target]
        """
        if query_positions.ndim != 3:
            raise ValueError(
                "query_positions must have shape [batch, queries, position_dim]"
            )
        if context_cells.ndim != 3:
            raise ValueError("context_cells must have shape [batch, cells, total_dim]")

        batch_size, num_queries, pos_dim = query_positions.shape
        context_batch_size, num_context_cells, context_dim = context_cells.shape

        if context_batch_size != batch_size:
            raise ValueError(
                "query_positions and context_cells must have same batch size"
            )
        if pos_dim != self.config.position_dim:
            raise ValueError(
                f"Expected query position dimension {self.config.position_dim}, "
                f"got {pos_dim}"
            )
        if context_dim != self.config.total_dim:
            raise ValueError(
                f"Expected context cell dimension {self.config.total_dim}, "
                f"got {context_dim}"
            )

        context_positions = self.positions_from_cells(context_cells)
        context_non_position = context_cells[..., self.config.position_dim :]

        # query_positions:   [B, Q, P] -> [B, Q, 1, P]
        # context_positions: [B, C, P] -> [B, 1, C, P]
        query_positions_expanded = query_positions.unsqueeze(2)
        context_positions_expanded = context_positions.unsqueeze(1)

        if self.config.displacement_convention == "other_minus_query":
            displacement_positions = (
                context_positions_expanded - query_positions_expanded
            )
        elif self.config.displacement_convention == "query_minus_other":
            displacement_positions = (
                query_positions_expanded - context_positions_expanded
            )
        else:
            raise ValueError(
                "Unknown displacement_convention: "
                f"{self.config.displacement_convention}"
            )

        context_non_position = context_non_position.unsqueeze(1).expand(
            -1,
            num_queries,
            -1,
            -1,
        )

        return torch.cat([displacement_positions, context_non_position], dim=-1)

    def aggregate_relative_context(
        self,
        relative_context: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encodes and aggregates relative context cells.

        relative_context: [batch, num_queries, num_context_cells, total_dim]
        pair_mask:        [batch, num_queries, num_context_cells], True for cells
                          that should contribute to the query aggregate.

        Returns:
            aggregate: [batch, num_queries, aggregate_dim]
        """
        if relative_context.ndim != 4:
            raise ValueError(
                "relative_context must have shape "
                "[batch, queries, context_cells, total_dim]"
            )
        if pair_mask.ndim != 3:
            raise ValueError(
                "pair_mask must have shape [batch, queries, context_cells]"
            )

        encoded = self.phi(relative_context)
        mask_float = pair_mask.unsqueeze(-1).to(encoded.dtype)
        encoded = encoded * mask_float

        aggregate = encoded.sum(dim=2)

        if self.config.aggregation == "mean":
            counts = mask_float.sum(dim=2).clamp_min(1.0)
            aggregate = aggregate / counts
        elif self.config.aggregation == "sum":
            pass
        else:
            raise ValueError(f"Unknown aggregation: {self.config.aggregation}")

        return aggregate

    def make_rho_input(
        self,
        query_positions: torch.Tensor,
        aggregate: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.use_query_position_in_rho:
            return torch.cat([query_positions, aggregate], dim=-1)
        return aggregate

    def forward(
        self,
        cells: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        cells: [batch, max_cells, total_dim]
        mask:  [batch, max_cells], True for valid cells, False for padding

        Returns:
            pred_targets: [batch, max_cells, target_dim]
        """
        if cells.ndim != 3:
            raise ValueError(
                f"Expected cells to have shape [batch, max_cells, total_dim], "
                f"got {tuple(cells.shape)}"
            )

        batch_size, max_cells, dim = cells.shape

        if dim != self.config.total_dim:
            raise ValueError(
                f"Expected last dimension {self.config.total_dim}, got {dim}"
            )

        if mask is None:
            mask = torch.ones(
                batch_size,
                max_cells,
                dtype=torch.bool,
                device=cells.device,
            )

        mask = mask.to(device=cells.device, dtype=torch.bool)
        positions = self.positions_from_cells(cells)

        # For query cell i and context cell j, phi sees
        # [position_j - position_i | additional_j | target_j].
        relative_context = self.relative_context_from_queries(
            query_positions=positions,
            context_cells=cells,
        )

        # pair_mask[b, i, j] is True exactly when:
        #   - i is a real query cell,
        #   - j is a real context cell,
        #   - j != i, so we get leave-one-out aggregation.
        valid_query = mask.unsqueeze(2)
        valid_context = mask.unsqueeze(1)
        not_self = ~torch.eye(
            max_cells,
            dtype=torch.bool,
            device=cells.device,
        ).unsqueeze(0)
        pair_mask = valid_query & valid_context & not_self

        aggregate_excluding_self = self.aggregate_relative_context(
            relative_context=relative_context,
            pair_mask=pair_mask,
        )

        rho_input = self.make_rho_input(
            query_positions=positions,
            aggregate=aggregate_excluding_self,
        )

        pred_targets = self.rho(rho_input)
        return pred_targets

    @torch.no_grad()
    def predict_at_positions(
        self,
        context_cells: torch.Tensor,
        query_positions: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict a movement field at arbitrary query positions.

        This is useful after training if you want to evaluate the learned velocity
        field on a spatial grid.

        For a query point q and context cell j, phi sees:
            [position_j - q | additional_j | target_j]

        Unlike forward(), this does not leave out any context cell, because the
        query positions are arbitrary field points rather than cells from the
        context set. To predict leave-one-out targets for actual cells in a frame,
        use forward() or predict_frame().

        context_cells:    [batch, num_context_cells, total_dim]
        query_positions:  [batch, num_query_points, position_dim]
        context_mask:     [batch, num_context_cells]

        Returns:
            pred_targets: [batch, num_query_points, target_dim]
        """
        if context_cells.ndim != 3:
            raise ValueError("context_cells must have shape [batch, cells, total_dim]")

        if query_positions.ndim != 3:
            raise ValueError(
                "query_positions must have shape [batch, query_points, position_dim]"
            )

        batch_size, num_context_cells, dim = context_cells.shape

        if dim != self.config.total_dim:
            raise ValueError(
                f"Expected context cell dimension {self.config.total_dim}, got {dim}"
            )

        if query_positions.shape[0] != batch_size:
            raise ValueError(
                "context_cells and query_positions must have same batch size"
            )

        if query_positions.shape[-1] != self.config.position_dim:
            raise ValueError(
                f"Expected query position dimension {self.config.position_dim}, "
                f"got {query_positions.shape[-1]}"
            )

        if context_mask is None:
            context_mask = torch.ones(
                batch_size,
                num_context_cells,
                dtype=torch.bool,
                device=context_cells.device,
            )

        context_mask = context_mask.to(device=context_cells.device, dtype=torch.bool)

        relative_context = self.relative_context_from_queries(
            query_positions=query_positions,
            context_cells=context_cells,
        )

        num_query_points = query_positions.shape[1]
        pair_mask = context_mask.unsqueeze(1).expand(
            -1,
            num_query_points,
            -1,
        )

        aggregate = self.aggregate_relative_context(
            relative_context=relative_context,
            pair_mask=pair_mask,
        )

        rho_input = self.make_rho_input(
            query_positions=query_positions,
            aggregate=aggregate,
        )
        return self.rho(rho_input)


# ============================================================
# Dataset and batching
# ============================================================


class CellSetDataset(Dataset):
    """
    Dataset of frames.

    Each item is one frame:
        frame: [num_cells_in_frame, total_dim]

    Each cell vector should be ordered as:
        [position coordinates | additional features | target vector]

    Example for position_dim=2, additional_dim=4, target_dim=2:

        [
            x, y,
            feature_1, feature_2, feature_3, feature_4,
            velocity_x, velocity_y,
        ]
    """

    def __init__(
        self,
        frames: Sequence[torch.Tensor],
        expected_dim: int = total_dim,
    ):
        self.frames: list[torch.Tensor] = []

        for idx, frame in enumerate(frames):
            frame_tensor = torch.as_tensor(frame, dtype=torch.float32)

            if frame_tensor.ndim != 2:
                raise ValueError(
                    f"Frame {idx} must have shape [num_cells, total_dim], "
                    f"got {tuple(frame_tensor.shape)}"
                )

            if frame_tensor.shape[1] != expected_dim:
                raise ValueError(
                    f"Frame {idx} has dimension {frame_tensor.shape[1]}, "
                    f"expected {expected_dim}"
                )

            if frame_tensor.shape[0] == 0:
                raise ValueError(f"Frame {idx} has no cells")

            self.frames.append(frame_tensor)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.frames[idx]


def collate_cell_sets(
    batch: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pads variable-size cell sets into a batch.

    Returns:
        cells: [batch, max_cells, total_dim]
        mask:  [batch, max_cells]
    """
    batch_size = len(batch)
    max_cells = max(frame.shape[0] for frame in batch)
    dim = batch[0].shape[1]

    cells = torch.zeros(batch_size, max_cells, dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_cells, dtype=torch.bool)

    for batch_idx, frame in enumerate(batch):
        num_cells = frame.shape[0]
        cells[batch_idx, :num_cells] = frame
        mask[batch_idx, :num_cells] = True

    return cells, mask


# ============================================================
# Training and evaluation
# ============================================================


def evaluate_model(
    model: LeaveOneOutDeepSets,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()

    total_loss = 0.0
    total_cells = 0

    with torch.no_grad():
        for cells, mask in loader:
            cells = cells.to(device)
            mask = mask.to(device)

            targets = model.targets_from_cells(cells)
            preds = model(cells, mask)

            per_cell_loss = (preds - targets).pow(2).mean(dim=-1)
            mask_float = mask.to(per_cell_loss.dtype)

            total_loss += (per_cell_loss * mask_float).sum().item()
            total_cells += int(mask.sum().item())

    return total_loss / max(total_cells, 1)


def train_model(
    model: LeaveOneOutDeepSets,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = epochs,
    lr: float = learning_rate,
    weight_decay: float = weight_decay,
    max_grad_norm: Optional[float] = max_grad_norm,
    device: Optional[torch.device | str] = None,
) -> list[dict[str, float]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_cells = 0

        for cells, mask in train_loader:
            cells = cells.to(device)
            mask = mask.to(device)

            targets = model.targets_from_cells(cells)
            preds = model(cells, mask)

            loss = masked_mse_loss(preds, targets, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            num_real_cells = int(mask.sum().item())
            total_loss += loss.item() * num_real_cells
            total_cells += num_real_cells

        train_loss = total_loss / max(total_cells, 1)

        log = {"epoch": float(epoch), "train_loss": train_loss}

        if val_loader is not None:
            val_loss = evaluate_model(model, val_loader, device)
            log["val_loss"] = val_loss
            print(
                f"Epoch {epoch:03d} | "
                f"train MSE {train_loss:.6f} | "
                f"val MSE {val_loss:.6f}"
            )
        else:
            print(f"Epoch {epoch:03d} | train MSE {train_loss:.6f}")

        history.append(log)

    return history


# ============================================================
# Prediction helpers
# ============================================================


@torch.no_grad()
def predict_frame(
    model: LeaveOneOutDeepSets,
    frame: torch.Tensor,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """
    Predict targets for every cell in a single frame using leave-one-out context.

    frame: [num_cells, total_dim]

    Returns:
        pred_targets: [num_cells, target_dim]
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    frame = torch.as_tensor(frame, dtype=torch.float32).to(device)

    cells = frame.unsqueeze(0)
    mask = torch.ones(1, frame.shape[0], dtype=torch.bool, device=device)

    preds = model(cells, mask)
    return preds[0].cpu()


@torch.no_grad()
def predict_field_at_positions(
    model: LeaveOneOutDeepSets,
    context_frame: torch.Tensor,
    query_positions: torch.Tensor,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """
    Predict the target vector at arbitrary spatial positions.

    context_frame:
        [num_cells, total_dim]

    query_positions:
        [num_query_points, position_dim]

    Returns:
        [num_query_points, target_dim]
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    context_frame = torch.as_tensor(context_frame, dtype=torch.float32).to(device)
    query_positions = torch.as_tensor(query_positions, dtype=torch.float32).to(device)

    context_cells = context_frame.unsqueeze(0)
    query_positions = query_positions.unsqueeze(0)

    context_mask = torch.ones(
        1,
        context_frame.shape[0],
        dtype=torch.bool,
        device=device,
    )

    preds = model.predict_at_positions(
        context_cells=context_cells,
        query_positions=query_positions,
        context_mask=context_mask,
    )

    return preds[0].cpu()


# ============================================================
# Synthetic data example
# Replace this with your real cell-track data.
# ============================================================


def make_synthetic_frames(
    num_frames: int = 800,
    min_cells: int = 8,
    max_cells: int = 30,
    noise_std: float = 0.03,
    seed: int = 0,
) -> list[torch.Tensor]:
    """
    Creates toy data shaped like:

        frame: [num_cells, position_dim + additional_dim + target_dim]

    This is only for testing that the model trains end-to-end.
    """
    generator = torch.Generator().manual_seed(seed)

    w_pos = (
        torch.randn(position_dim, target_dim, generator=generator)
        / max(position_dim, 1) ** 0.5
    )
    w_add = (
        torch.randn(additional_dim, target_dim, generator=generator)
        / max(additional_dim, 1) ** 0.5
    )
    w_global = (
        torch.randn(
            position_dim + additional_dim,
            target_dim,
            generator=generator,
        )
        / max(position_dim + additional_dim, 1) ** 0.5
    )

    frames: list[torch.Tensor] = []

    for _ in range(num_frames):
        num_cells = int(
            torch.randint(
                low=min_cells,
                high=max_cells + 1,
                size=(1,),
                generator=generator,
            ).item()
        )

        positions = 2.0 * torch.rand(num_cells, position_dim, generator=generator) - 1.0
        additional = torch.randn(num_cells, additional_dim, generator=generator)

        no_target_context = torch.cat([positions, additional], dim=-1)

        global_feature = torch.tanh(
            no_target_context.mean(dim=0, keepdim=True) @ w_global
        )

        target = positions @ w_pos

        if additional_dim > 0:
            target = target + torch.tanh(additional @ w_add)

        target = target + global_feature.expand(num_cells, -1)
        target = target + noise_std * torch.randn(
            num_cells,
            target_dim,
            generator=generator,
        )

        frame = torch.cat([positions, additional, target], dim=-1)
        frames.append(frame)

    return frames


# ============================================================
# Main example
# ============================================================

if __name__ == "__main__":
    config = DeepSetsConfig(
        position_dim=position_dim,
        additional_dim=additional_dim,
        target_dim=target_dim,
        aggregate_dim=aggregate_dim,
        phi_hidden_dims=phi_hidden_dims,
        rho_hidden_dims=rho_hidden_dims,
        aggregation=aggregation,
        displacement_convention=displacement_convention,
        use_query_position_in_rho=use_query_position_in_rho,
        dropout=dropout,
    )

    model = LeaveOneOutDeepSets(config)

    # Replace this with your real data:
    #
    # frames = [
    #     frame_0,  # shape [num_cells_0, total_dim]
    #     frame_1,  # shape [num_cells_1, total_dim]
    #     ...
    # ]
    #
    # Each row should be:
    #     [position | additional features | target]
    #
    frames = make_synthetic_frames(num_frames=800)

    split_idx = int(0.8 * len(frames))
    train_frames = frames[:split_idx]
    val_frames = frames[split_idx:]

    train_dataset = CellSetDataset(train_frames, expected_dim=config.total_dim)
    val_dataset = CellSetDataset(val_frames, expected_dim=config.total_dim)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_cell_sets,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_cell_sets,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
    )

    # Predict target vectors for every cell in one frame.
    sample_frame = val_frames[0]
    predicted_targets = predict_frame(model, sample_frame, device=device)

    print("Sample frame shape:", tuple(sample_frame.shape))
    print("Predicted targets shape:", tuple(predicted_targets.shape))

    # Example: predict a learned movement field on arbitrary positions.
    # This only makes direct spatial sense when position_dim == 2.
    if position_dim == 2:
        xs = torch.linspace(-1.0, 1.0, 20)
        ys = torch.linspace(-1.0, 1.0, 20)

        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        query_positions = torch.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1)],
            dim=-1,
        )

        field_predictions = predict_field_at_positions(
            model=model,
            context_frame=sample_frame,
            query_positions=query_positions,
            device=device,
        )

        print("Query positions shape:", tuple(query_positions.shape))
        print("Field predictions shape:", tuple(field_predictions.shape))
