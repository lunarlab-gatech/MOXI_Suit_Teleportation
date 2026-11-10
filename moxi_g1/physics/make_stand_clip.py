"""A one-clip G1 motion subset (the neutral stand, held), so that
`deploy.sim2sim.export_assets` can write the G1 sim scene without the SEED dataset.

export_assets builds MorphMimic's eval env, which needs a motion library, and then
writes `<out>/g1.xml` (the training scene as plain MJCF) plus the clips asked for. For
teleop we only need the scene; the clip here is `stand.json["g1"]["state"]` (one full
reference state) repeated for 5 s, which also plays as a "stand still" test clip.

From `tracking/morphmimic/`:

  uv run python ../../moxi_g1/physics/make_stand_clip.py     # writes moxi_g1/physics/data/
  uv run python -m deploy.sim2sim.export_assets --robot g1 \
    --subset-yaml ../../moxi_g1/physics/data/stand_subset.yaml --clips moxi_stand \
    --out ../../moxi_g1/physics/assets
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
MORPHMIMIC = HERE.parents[1] / "tracking" / "morphmimic"
OUT = HERE / "data"
SECONDS = 5.0
FPS = 50


def main() -> None:
  robot = json.loads((MORPHMIMIC / "deploy/sim2sim/robots/g1.json").read_text())
  stand = json.loads((MORPHMIMIC / "deploy/vr_teleop/gmr_configs/stand.json").read_text())["g1"]
  state = np.asarray(stand["state"], np.float64)
  joints, bodies = robot["canonical_joint_names"], robot["tracked_bodies"]
  nj, nb = len(joints), len(bodies)
  q = state[:nj]
  pos = state[2 * nj: 2 * nj + 3 * nb].reshape(nb, 3)
  quat = state[2 * nj + 3 * nb: 2 * nj + 7 * nb].reshape(nb, 4)
  t = int(SECONDS * FPS)
  OUT.mkdir(parents=True, exist_ok=True)
  np.savez(
    OUT / "moxi_stand.npz",
    fps=FPS,
    joint_names=np.array(joints),
    body_names=np.array(bodies),
    joint_pos=np.tile(q, (t, 1)).astype(np.float32),
    joint_vel=np.zeros((t, nj), np.float32),
    body_pos_w=np.tile(pos, (t, 1, 1)).astype(np.float32),
    body_quat_w=np.tile(quat, (t, 1, 1)).astype(np.float32),
    body_lin_vel_w=np.zeros((t, nb, 3), np.float32),
    body_ang_vel_w=np.zeros((t, nb, 3), np.float32),
  )
  with open(OUT / "stand_subset.yaml", "w") as f:
    yaml.safe_dump({"root": str(OUT), "hours": SECONDS / 3600, "clips": ["moxi_stand.npz"]},
                   f, sort_keys=False)
  print(f"wrote {OUT / 'moxi_stand.npz'} ({t} frames) and {OUT / 'stand_subset.yaml'}")


if __name__ == "__main__":
  main()
