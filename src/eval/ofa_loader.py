from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

DEFAULT_OFA_ROOT = Path("data/external/once-for-all")
DEFAULT_NET_ID = "ofa_mbv3_d234_e346_k357_w1.0"


def _ensure_ofa_on_path(ofa_root: Path = DEFAULT_OFA_ROOT) -> None:
    if not ofa_root.exists():
        raise FileNotFoundError(
            f"OFA repo not found at {ofa_root}. Clone https://github.com/mit-han-lab/once-for-all.git there."
        )
    ofa_path = str(ofa_root.resolve())
    if ofa_path not in sys.path:
        sys.path.insert(0, ofa_path)


def load_ofa_supernet(
    checkpoint_path: Path,
    net_id: str = DEFAULT_NET_ID,
    ofa_root: Path = DEFAULT_OFA_ROOT,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load the OFA supernet and move to device."""
    _ensure_ofa_on_path(ofa_root)
    from ofa.model_zoo import ofa_net  # type: ignore

    net = ofa_net(net_id, pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)
    if device:
        net.to(device)
    net.eval()
    return net


def load_ofa_net(config: Dict[str, Any], checkpoint_path: Optional[Path] = None, device: str = "cuda") -> torch.nn.Module:
    net_id = config.get("net_id", DEFAULT_NET_ID)
    ckpt = checkpoint_path or Path("data/checkpoints/ofa") / f"{net_id}.pth"
    dev = torch.device(device)
    return load_ofa_supernet(ckpt, net_id=net_id, device=dev)


def get_net_for_config(
    cfg: Dict,
    checkpoint_path: Path,
    net_id: str = DEFAULT_NET_ID,
    ofa_root: Path = DEFAULT_OFA_ROOT,
    device: torch.device | None = None,
) -> Tuple[torch.nn.Module, int]:
    """
    Build an active sub-network for the given discrete config.

    Expected cfg keys: kernel_size, expand_ratio, depth, image_size.
    """
    net = load_ofa_supernet(checkpoint_path, net_id=net_id, ofa_root=ofa_root, device=device)

    ks = int(cfg.get("kernel_size", 3))
    e = int(cfg.get("expand_ratio", 4))
    d = int(cfg.get("depth", 3))

    # Apply a uniform subnet activation; OFA will broadcast these choices per stage.
    net.set_active_subnet(ks=ks, e=e, d=d)
    active_subnet = net.get_active_subnet(preserve_weight=True)
    if device:
        active_subnet.to(device)
    active_subnet.eval()

    image_size = int(cfg.get("image_size", 224))
    return active_subnet, image_size


def make_image_size(config: Dict[str, Any]) -> int:
    return int(max(32, config.get("image_size", 224)))