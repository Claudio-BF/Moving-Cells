"""SO(2)-equivariant cell-velocity prediction in 3D.

Training rows are [x, y, z, brightness, vx, vy, vz].
Inference rows may omit the target: [x, y, z, brightness].

Dependencies:
    pip install torch torch-geometric lightning scipy
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, pi, sin
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from scipy.spatial import cKDTree
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, MessagePassing
from torch_geometric.utils import degree


@dataclass(frozen=True)
class ModelConfig:
    radius: float
    hidden_dim: int = 64
    layers: int = 3
    max_neighbors: int = 48
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.radius <= 0 or self.hidden_dim < 1 or self.layers < 1:
            raise ValueError("radius, hidden_dim, and layers must be positive")
        if self.max_neighbors < 1 or not 0 <= self.dropout < 1:
            raise ValueError(
                "max_neighbors must be positive and dropout must be in [0, 1)"
            )


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    gradient_clip: float = 1.0
    num_workers: int = 0


@dataclass(frozen=True)
class DataStats:
    """Rotation-compatible normalization fitted on training data only."""

    brightness_mean: float
    brightness_std: float
    velocity_xy_scale: float
    velocity_z_scale: float

    @classmethod
    def fit(cls, frames: Sequence[Tensor | np.ndarray]) -> "DataStats":
        if len(frames) == 0:
            raise ValueError("At least one training frame is required")
        rows = torch.cat([_as_frame(frame, require_target=True) for frame in frames])
        brightness, velocity = rows[:, 3], rows[:, 4:7]
        eps = torch.finfo(rows.dtype).eps
        return cls(
            brightness_mean=brightness.mean().item(),
            brightness_std=brightness.std(unbiased=False).clamp_min(eps).item(),
            # One shared x/y scale is essential: separate scales would break SO(2).
            velocity_xy_scale=velocity[:, :2]
            .square()
            .mean()
            .sqrt()
            .clamp_min(eps)
            .item(),
            velocity_z_scale=velocity[:, 2]
            .square()
            .mean()
            .sqrt()
            .clamp_min(eps)
            .item(),
        )

    @property
    def velocity_scale(self) -> Tensor:
        return torch.tensor(
            [self.velocity_xy_scale, self.velocity_xy_scale, self.velocity_z_scale]
        )

    def decode_velocity(self, velocity: Tensor) -> Tensor:
        return velocity * self.velocity_scale.to(velocity)


def _as_frame(frame: Tensor | np.ndarray, *, require_target: bool) -> Tensor:
    frame = torch.as_tensor(frame, dtype=torch.float32)
    valid_dims = {7} if require_target else {4, 7}
    if frame.ndim != 2 or frame.shape[1] not in valid_dims or len(frame) == 0:
        expected = "[cells, 7]" if require_target else "[cells, 4] or [cells, 7]"
        raise ValueError(f"Expected {expected}; got {tuple(frame.shape)}")
    if not torch.isfinite(frame).all():
        raise ValueError("Frames cannot contain NaN or infinity")
    return frame


def radius_edges(pos: Tensor, radius: float, max_neighbors: int) -> Tensor:
    """Build directed j->i radius edges once, using SciPy's optimized KD-tree."""
    if radius <= 0 or max_neighbors < 1:
        raise ValueError("radius and max_neighbors must be positive")

    n = len(pos)
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long)

    points = pos.detach().cpu().numpy()
    k = min(max_neighbors + 1, n)  # +1 because a point normally finds itself.
    distances, neighbors = cKDTree(points).query(
        points, k=k, distance_upper_bound=radius
    )
    if k == 1:
        distances, neighbors = distances[:, None], neighbors[:, None]

    target = np.repeat(np.arange(n), k)
    source = neighbors.reshape(-1)
    valid = np.isfinite(distances.reshape(-1)) & (source < n) & (source != target)
    return torch.from_numpy(np.stack([source[valid], target[valid]])).long()


def make_graph(
    frame: Tensor | np.ndarray,
    stats: DataStats,
    config: ModelConfig,
) -> Data:
    """Convert one variable-size frame to a sparse PyG graph."""
    frame = _as_frame(frame, require_target=False)
    pos = frame[:, :3]
    brightness = (frame[:, 3:4] - stats.brightness_mean) / stats.brightness_std
    data = Data(
        x=brightness,
        pos=pos,
        edge_index=radius_edges(pos, config.radius, config.max_neighbors),
        num_nodes=len(frame),
    )
    if frame.shape[1] == 7:
        data.y = frame[:, 4:7] / stats.velocity_scale
    return data


def make_graphs(
    frames: Sequence[Tensor | np.ndarray],
    stats: DataStats,
    config: ModelConfig,
) -> list[Data]:
    return [make_graph(frame, stats, config) for frame in frames]


def _edge_geometry(
    pos_i: Tensor, pos_j: Tensor, radius: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Return relative vectors, SO(2)-invariants, and a smooth radial cutoff."""
    relative = pos_j - pos_i
    rho = relative[:, :2].norm(dim=-1, keepdim=True)
    distance = relative.norm(dim=-1, keepdim=True)
    invariants = torch.cat([rho, relative[:, 2:3], distance], dim=-1) / radius
    cutoff = 0.5 * (torch.cos(pi * (distance / radius).clamp(max=1.0)) + 1.0)
    return relative, invariants, cutoff


class ScalarMessageBlock(MessagePassing):
    """EGNN-style message passing whose hidden features are SO(2)-invariant scalars."""

    def __init__(self, hidden_dim: int, radius: float, dropout: float):
        super().__init__(aggr="mean")
        self.radius = radius
        self.message_mlp = MLP(
            [2 * hidden_dim + 3, hidden_dim, hidden_dim],
            act="silu",
            norm=None,
            dropout=float(dropout),
        )
        self.update_mlp = MLP(
            [2 * hidden_dim, hidden_dim, hidden_dim],
            act="silu",
            norm=None,
            dropout=float(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        messages = self.propagate(edge_index, h=h, pos=pos)
        return self.norm(h + self.update_mlp(torch.cat([h, messages], dim=-1)))

    def message(self, h_i: Tensor, h_j: Tensor, pos_i: Tensor, pos_j: Tensor) -> Tensor:
        _, invariants, cutoff = _edge_geometry(pos_i, pos_j, self.radius)
        return cutoff * self.message_mlp(torch.cat([h_i, h_j, invariants], dim=-1))


class EquivariantVelocityReadout(MessagePassing):
    """Construct velocity from radial, tangential, and z-axis equivariant bases."""

    def __init__(self, hidden_dim: int, radius: float, dropout: float):
        super().__init__(aggr="mean")
        self.radius = radius
        self.coefficient_mlp = MLP(
            [2 * hidden_dim + 3, hidden_dim, 3],
            act="silu",
            norm=None,
            dropout=float(dropout),
        )
        # A scalar can contribute directly to v_z. A self-only xy vector
        # would violate SO(2) equivariance.
        self.self_z = MLP(
            [hidden_dim, hidden_dim, 1], act="silu", norm=None, dropout=float(dropout)
        )

    def forward(self, h: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        velocity = self.propagate(edge_index, h=h, pos=pos)
        return velocity + F.pad(self.self_z(h), (2, 0))

    def message(self, h_i: Tensor, h_j: Tensor, pos_i: Tensor, pos_j: Tensor) -> Tensor:
        relative, invariants, cutoff = _edge_geometry(pos_i, pos_j, self.radius)
        coefficients = cutoff * self.coefficient_mlp(
            torch.cat([h_i, h_j, invariants], dim=-1)
        )

        radial = F.normalize(relative[:, :2], dim=-1, eps=1e-8)
        tangential = torch.stack([-radial[:, 1], radial[:, 0]], dim=-1)
        xy = coefficients[:, 0:1] * radial + coefficients[:, 1:2] * tangential
        return torch.cat([xy, coefficients[:, 2:3]], dim=-1)


class AxialVelocityGNN(nn.Module):
    """Translation-invariant and exactly SO(2)-equivariant about the z-axis."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = MLP(
            [2, config.hidden_dim, config.hidden_dim],
            act="silu",
            norm=None,
            dropout=float(config.dropout),
        )
        self.blocks = nn.ModuleList(
            ScalarMessageBlock(config.hidden_dim, config.radius, config.dropout)
            for _ in range(config.layers)
        )
        self.readout = EquivariantVelocityReadout(
            config.hidden_dim, config.radius, config.dropout
        )

    def forward(self, brightness: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        local_degree = degree(
            edge_index[1], num_nodes=len(pos), dtype=pos.dtype
        ).unsqueeze(-1)
        local_density = torch.log1p(local_degree) / torch.log1p(
            pos.new_tensor(float(self.config.max_neighbors))
        )
        h = self.encoder(torch.cat([brightness, local_density], dim=-1))
        for block in self.blocks:
            h = block(h, pos, edge_index)
        return self.readout(h, pos, edge_index)


class VelocityTask(L.LightningModule):
    def __init__(
        self,
        model_config: dict,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AxialVelocityGNN(ModelConfig(**model_config))

    def forward(self, batch: Data) -> Tensor:
        return self.model(batch.x, batch.pos, batch.edge_index)

    def _step(self, batch: Data, stage: str) -> Tensor:
        loss = F.mse_loss(self(batch), batch.y)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_nodes,
        )
        return loss

    def training_step(self, batch: Data, _: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Data, _: int) -> None:
        self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


def fit(
    train_frames: Sequence[Tensor | np.ndarray],
    val_frames: Sequence[Tensor | np.ndarray],
    model_config: ModelConfig,
    train_config: TrainConfig = TrainConfig(),
) -> tuple[AxialVelocityGNN, DataStats]:
    """Fit with automatic batching, early stopping, LR scheduling, and best-checkpoint restore."""
    if len(train_frames) == 0 or len(val_frames) == 0:
        raise ValueError("train_frames and val_frames must both be non-empty")
    L.seed_everything(0, workers=True)
    stats = DataStats.fit(train_frames)
    loader_options = dict(
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        persistent_workers=train_config.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(
        make_graphs(train_frames, stats, model_config),
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        make_graphs(val_frames, stats, model_config),
        **loader_options,
    )

    with TemporaryDirectory() as checkpoint_dir:
        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
        )
        task = VelocityTask(
            asdict(model_config),
            train_config.learning_rate,
            train_config.weight_decay,
        )
        trainer = L.Trainer(
            max_epochs=train_config.max_epochs,
            accelerator="auto",
            devices=1,
            gradient_clip_val=train_config.gradient_clip,
            callbacks=[
                EarlyStopping("val_loss", mode="min", patience=train_config.patience),
                checkpoint,
            ],
            logger=False,
            enable_progress_bar=True,
        )
        trainer.fit(task, train_loader, val_loader)
        best_task = VelocityTask.load_from_checkpoint(checkpoint.best_model_path)

    return best_task.model.eval(), stats


@torch.inference_mode()
def predict(
    model: AxialVelocityGNN,
    frame: Tensor | np.ndarray,
    stats: DataStats,
    device: str | torch.device | None = None,
) -> Tensor:
    """Predict physical-unit velocities from [x,y,z,brightness] rows."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    graph = make_graph(frame, stats, model.config).to(device)
    model = model.to(device).eval()
    normalized = model(graph.x, graph.pos, graph.edge_index)
    return stats.decode_velocity(normalized).cpu()


def rotate_z(vectors: Tensor, angle: float) -> Tensor:
    rotation = vectors.new_tensor(
        [[cos(angle), -sin(angle), 0.0], [sin(angle), cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    return vectors @ rotation.T


@torch.inference_mode()
def equivariance_error(
    model: AxialVelocityGNN, graph: Data, angle: float = 0.73
) -> float:
    """Numerically verify f(Rp,b)=R f(p,b) on one graph."""
    model = model.eval().to(graph.pos.device)
    prediction = model(graph.x, graph.pos, graph.edge_index)
    rotated_prediction = model(graph.x, rotate_z(graph.pos, angle), graph.edge_index)
    expected = rotate_z(prediction, angle)
    return (rotated_prediction - expected).abs().max().item()


def save_model(path: str | Path, model: AxialVelocityGNN, stats: DataStats) -> None:
    torch.save(
        {
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "data_stats": asdict(stats),
        },
        path,
    )


def load_model(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[AxialVelocityGNN, DataStats]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model = AxialVelocityGNN(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), DataStats(**checkpoint["data_stats"])


if __name__ == "__main__":
    # Replace these with your own trajectory/experiment-level split.
    # train_frames = [...]  # each [N, 7]
    # val_frames = [...]    # each [N, 7]
    # config = ModelConfig(radius=<physical interaction radius>)
    # model, stats = fit(train_frames, val_frames, config)
    # velocities = predict(model, new_frame[:, :4], stats)
    raise SystemExit(
        "Import this module and provide train_frames and val_frames; "
        "see the example above."
    )
