#!/usr/bin/env python3
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory


class ActorMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(48, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 12),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PolicyRunner:
    def __init__(
        self,
        *,
        default_joint_angles: Sequence[float],
        hard_lower_limits: Sequence[float],
        hard_upper_limits: Sequence[float],
        policy_order_indices: Sequence[int],
        model_path: str | None = None,
        action_scale: float = 0.25,
        use_cuda: bool = True,
    ) -> None:
        if len(default_joint_angles) != 12:
            raise ValueError("default_joint_angles must contain 12 values")
        if len(hard_lower_limits) != 12:
            raise ValueError("hard_lower_limits must contain 12 values")
        if len(hard_upper_limits) != 12:
            raise ValueError("hard_upper_limits must contain 12 values")
        if len(policy_order_indices) != 12:
            raise ValueError("policy_order_indices must contain 12 indices")
        if action_scale <= 0.0:
            raise ValueError("action_scale must be positive")

        self.default_joint_angles = [float(value) for value in default_joint_angles]
        self.hard_lower_limits = [float(value) for value in hard_lower_limits]
        self.hard_upper_limits = [float(value) for value in hard_upper_limits]
        self.policy_order_indices = [int(value) for value in policy_order_indices]
        self.action_scale = float(action_scale)
        self.model_path = self._resolve_model_path(model_path)
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.model = self._load_model(self.model_path, self.device)
        self._default_joint_angles_policy = [
            self.default_joint_angles[base_idx] for base_idx in self.policy_order_indices
        ]
        self._last_action = [0.0] * 12
        self._target_joint_angles = list(self.default_joint_angles)

    @property
    def device_type(self) -> str:
        return self.device.type

    @staticmethod
    def resolve_default_model_path() -> Path:
        try:
            return (
                Path(get_package_share_directory("policy_action_controller"))
                / "model"
                / "model_3950.pt"
            )
        except PackageNotFoundError:
            return Path(__file__).resolve().parents[1] / "model" / "model_3950.pt"

    def infer_action(self, observation: Sequence[float]) -> list[float]:
        if len(observation) != 48:
            raise ValueError("observation must contain 48 values")
        obs_tensor = torch.tensor(
            [float(value) for value in observation],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            action = self.model(obs_tensor).squeeze(0)
        return [float(value) for value in action.detach().cpu().tolist()]

    def get_last_action(self) -> list[float]:
        return list(self._last_action)

    def get_target_joint_angles(self) -> list[float]:
        return list(self._target_joint_angles)

    def reset_state(self) -> None:
        self._last_action = [0.0] * 12
        self._target_joint_angles = list(self.default_joint_angles)

    def action_to_target_joint_angles(
        self, action: Sequence[float]
    ) -> tuple[list[float], list[float]]:
        if len(action) != 12:
            raise ValueError("action must contain 12 values")

        clipped_action = [max(-1.0, min(1.0, float(value))) for value in action]
        target_joint_angles = list(self.default_joint_angles)
        for policy_idx, base_idx in enumerate(self.policy_order_indices):
            if base_idx < 0 or base_idx >= 12:
                raise ValueError("policy_order_indices must contain indices in [0, 11]")
            target_joint_angles[base_idx] = float(
                self.default_joint_angles[base_idx] + self.action_scale * clipped_action[policy_idx]
            )
            target_joint_angles[base_idx] = max(
                self.hard_lower_limits[base_idx],
                min(self.hard_upper_limits[base_idx], target_joint_angles[base_idx]),
            )

        self._last_action = list(clipped_action)
        self._target_joint_angles = target_joint_angles
        return list(self._target_joint_angles), list(clipped_action)

    def infer_target_joint_angles(
        self, observation: Sequence[float]
    ) -> tuple[list[float], list[float]]:
        raw_action = self.infer_action(observation)
        return self.action_to_target_joint_angles(raw_action)

    def _resolve_model_path(self, model_path: str | None) -> Path:
        if model_path:
            path = Path(model_path).expanduser().resolve()
        else:
            path = self.resolve_default_model_path().expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return path

    def _load_model(self, model_path: Path, device: torch.device) -> nn.Module:
        ckpt = torch.load(model_path.as_posix(), map_location=device)
        model = ActorMLP()
        state_dict = self._extract_actor_state_dict(ckpt)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        return model

    def _extract_actor_state_dict(self, ckpt: object) -> dict[str, torch.Tensor]:
        if isinstance(ckpt, dict) and "actor_state_dict" in ckpt:
            raw_state_dict = ckpt["actor_state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            raw_state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            raw_state_dict = ckpt
        else:
            raise TypeError(f"Unsupported checkpoint type: {type(ckpt).__name__}")

        if not isinstance(raw_state_dict, dict):
            raise TypeError("Actor state dict must be a mapping of parameter names to tensors")

        normalized_state_dict: dict[str, torch.Tensor] = {}
        unsupported_keys: list[str] = []
        for key, value in raw_state_dict.items():
            if not isinstance(key, str):
                unsupported_keys.append(str(key))
                continue
            if key.startswith("distribution."):
                continue
            if key.startswith("mlp."):
                normalized_key = key
            elif key.startswith("actor."):
                normalized_key = "mlp." + key[len("actor.") :]
            else:
                unsupported_keys.append(key)
                continue
            normalized_state_dict[normalized_key] = value

        if unsupported_keys:
            raise ValueError(
                "Unsupported actor checkpoint keys: "
                + ", ".join(sorted(unsupported_keys))
            )

        expected_keys = set(ActorMLP().state_dict().keys())
        actual_keys = set(normalized_state_dict.keys())
        missing_keys = sorted(expected_keys - actual_keys)
        unexpected_keys = sorted(actual_keys - expected_keys)
        if missing_keys or unexpected_keys:
            details: list[str] = []
            if missing_keys:
                details.append("missing keys: " + ", ".join(missing_keys))
            if unexpected_keys:
                details.append("unexpected keys: " + ", ".join(unexpected_keys))
            raise ValueError("Actor checkpoint does not match inference model: " + "; ".join(details))

        return normalized_state_dict
