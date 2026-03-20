#!/usr/bin/env python3
from __future__ import annotations


def create_state_adapter(node, control_cfg: dict[str, object]):
    mode = str(control_cfg['mode'])
    if mode == 'sim2real':
        from sim2real_backend import Sim2RealStateAdapter

        return Sim2RealStateAdapter(node, control_cfg)

    from sim2sim_backend import Sim2SimStateAdapter

    return Sim2SimStateAdapter(node, control_cfg)


def create_control_backend(node, control_cfg: dict[str, object]):
    mode = str(control_cfg['mode'])
    if mode == 'sim2real':
        from sim2real_backend import Sim2RealControlBackend

        return Sim2RealControlBackend(node, control_cfg)

    from sim2sim_backend import Sim2SimControlBackend

    return Sim2SimControlBackend(node, control_cfg)
