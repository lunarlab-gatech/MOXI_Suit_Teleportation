"""MOXI Player (G1 avatar) -> MorphMimic's G1 reference stream, for `deploy.run --teleop`.

J-MEX's `jmex_x2_bridge` streams MOXI's retargeted *X2* to MorphMimic. This is the same
streamer for the *G1*: MOXI Player retargets you to `g1_three_fingers`, and each frame
becomes MorphMimic's 240-value G1 reference state, published as a `RefMessage` on ZMQ
:5556, which the policy node reads with `--robot g1 --teleop --interface motion`.

Everything except the robot comes from the bridge unchanged (the MOXI receiver, the
keys, 60 -> 50 Hz resampling, the OFF/IDLE/POSE modes, recording and replay, the ZMQ
publisher, the CLI). Only the X2-specific parts are swapped for G1 ones:
- joints: MOXI's G1 joints drive the G1's 29 native joints by name (identical to the
  canonical layout), clamped to the model's ranges; joints MOXI does not send hold the
  G1 neutral stand (`stand.json["g1"]`); finger joints are ignored;
- FK: GMR's G1 model (`third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml`), the one
  MorphMimic's own G1 streamer builds references with, for the 14 tracked bodies of
  `deploy/sim2sim/robots/g1.json`;
- calibration height: the G1 stand's `ankle_z`;
- the message's robot is "g1" (the policy node also accepts it for `--robot g1_amber`).
MOXI's root ("hip") frame is taken as the G1 pelvis (`--hip-to-pelvis`, default
identity): MOXI's G1 URDF and the G1 MJCF were checked to have the same kinematics.

Run inside the MorphMimic checkout's venv, from `tracking/morphmimic/` (see the root
README.md):

  uv run --group export --group teleop python ../../moxi_g1/physics/moxi_g1_streamer.py --channel 4
  ... --replay session.npz          # a MOXI session recorded with --record

Keys (no Enter): c calibrate (stand straight), space IDLE <-> POSE, x OFF, Ctrl-C quit.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]  # moxi_g1/physics/ -> the repo root
MORPHMIMIC = REPO / "tracking" / "morphmimic"
BRIDGE_SRC = REPO / "vendor" / "jmex_x2_bridge-0.1.0rc1" / "jmex_x2_bridge" / "src"

# The bridge resolves the checkout at import time.
os.environ.setdefault("MORPHMIMIC_DIR", str(MORPHMIMIC))
for p in (str(MORPHMIMIC), str(BRIDGE_SRC)):
  if p not in sys.path:
    sys.path.insert(0, p)

from deploy.sim2sim.obs import Robot  # noqa: E402  (MorphMimic)
from jmex_x2_bridge.contract.stream_constants import FPS  # noqa: E402
from jmex_x2_bridge.moxi import JointMap, JointTable, MoxiFrame, map_joints  # noqa: E402
from jmex_x2_bridge.streamer import cli, core  # noqa: E402
from jmex_x2_bridge.streamer.modes import Modes  # noqa: E402
from jmex_x2_bridge.streamer.retarget import IDENTITY, Pose, RetargetOutput  # noqa: E402

ROBOT_NAME = "g1"
G1 = Robot.load(ROBOT_NAME)
G1_XML = MORPHMIMIC / "third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml"
STAND = json.loads((MORPHMIMIC / "deploy/vr_teleop/gmr_configs/stand.json").read_text())[G1.body]
NATIVE: tuple[str, ...] = G1.native_joint_names
TRACKED: tuple[str, ...] = G1.tracked_bodies
ANKLES = [TRACKED.index(f"{s}_ankle_roll_link") for s in ("left", "right")]


def g1_joint_map(table: JointTable) -> JointMap:
  """MOXI's joints matched against the G1's 29 native joints (`x2_idx` indexes NATIVE)."""
  return map_joints(table, NATIVE)


class G1RetargetExtractor:
  """MOXI frames -> `RetargetOutput` (q: the G1's 29 native joints) for one joint table."""

  def __init__(self, table: JointTable, model: mujoco.MjModel,
               hip_to_pelvis: Pose = IDENTITY) -> None:
    self._offset_pos = np.asarray(hip_to_pelvis.pos, np.float64)
    self._offset_quat = np.asarray(hip_to_pelvis.quat, np.float64)
    self.joint_map = g1_joint_map(table)
    if self.joint_map.root is None:
      raise ValueError("MOXI sent no root joint: no root pose for the reference")
    self._root = self.joint_map.root
    self._moxi_idx = self.joint_map.moxi_idx
    self._g1_idx = self.joint_map.x2_idx
    self._lo, self._hi = model.jnt_range[[model.joint(n).id for n in NATIVE]].T
    self._stand = np.array([STAND["joint_pos"][n] for n in NATIVE], np.float64)

  def __call__(self, frame: MoxiFrame) -> RetargetOutput:
    q = self._stand.copy()
    q[self._g1_idx] = frame.angle[self._moxi_idx]
    root = frame.body_pos[self._root]
    xy = frame.base_pos[:2] if frame.has_base_motion else root[:2]
    hip_pos = np.array([xy[0], xy[1], root[2]], np.float64)
    hip_quat = np.asarray(frame.root_quat, np.float64)
    shift, quat = np.zeros(3), np.zeros(4)
    mujoco.mju_rotVecQuat(shift, self._offset_pos, hip_quat)
    mujoco.mju_mulQuat(quat, hip_quat, self._offset_quat)
    return RetargetOutput(q=np.clip(q, self._lo, self._hi), root_pos=hip_pos + shift,
                          root_quat=quat)


class G1ReferenceBuilder:
  """Reference states [240] of consecutive 50 Hz frames on the G1 model: canonical q and
  its backward difference, then the tracked bodies' FK poses and velocities (world)."""

  def __init__(self, model: mujoco.MjModel) -> None:
    self._m = model
    self._d = mujoco.MjData(model)
    self._qadr = model.jnt_qposadr[[model.joint(n).id for n in NATIVE]]
    self._bodies = [model.body(b).id for b in TRACKED]
    self._prev: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

  def _fk(self, out: RetargetOutput) -> tuple[np.ndarray, np.ndarray]:
    d = self._d
    d.qpos[0:3], d.qpos[3:7] = out.root_pos, out.root_quat
    d.qpos[self._qadr] = out.q
    mujoco.mj_kinematics(self._m, d)
    return d.xpos[self._bodies].copy(), d.xquat[self._bodies].copy()

  def ankle_z(self, out: RetargetOutput) -> float:
    pos, _ = self._fk(out)
    return float(pos[ANKLES, 2].mean())

  def reset(self) -> None:
    self._prev = None

  def state(self, out: RetargetOutput, z_offset: float = 0.0) -> np.ndarray:
    q_canon = G1.to_canonical(out.q)
    pos, quat = self._fk(out)
    pos[:, 2] += z_offset
    if self._prev is None:
      qd, lin, ang = np.zeros_like(q_canon), np.zeros_like(pos), np.zeros_like(pos)
    else:
      q0, pos0, quat0 = self._prev
      qd = (q_canon - q0) * FPS
      lin = (pos - pos0) * FPS
      ang = np.zeros_like(pos)
      turn, inv = np.zeros(4), np.zeros(4)
      for b in range(len(quat)):
        mujoco.mju_negQuat(inv, quat0[b])
        mujoco.mju_mulQuat(turn, quat[b], inv)  # World frame: q_t * q_{t-1}^-1.
        mujoco.mju_quat2Vel(ang[b], turn, 1.0 / FPS)
    self._prev = (q_canon, pos, quat)
    return np.concatenate((q_canon, qd, pos.ravel(), quat.ravel(), lin.ravel(), ang.ravel()))


class G1Modes(Modes):
  """The bridge's OFF/IDLE/POSE machine, calibrating to the G1 stand's ankle height."""

  def __init__(self) -> None:
    super().__init__()
    self._stand_ankle_z = float(STAND["ankle_z"])


def g1_joint_report(jm: JointMap) -> list[str]:
  driven = len(jm.x2_idx)
  return [
    f"joints: MOXI drives {driven} of the {len(NATIVE)} G1 joints (root: MOXI joint {jm.root})",
    "  ignored MOXI joints (no G1 joint): " + (", ".join(jm.unmatched_moxi) or "none"),
    "  G1 joints held at the stand (no MOXI joint): " + (", ".join(jm.uncovered_x2) or "none"),
  ]


def install() -> None:
  """Swap the bridge's X2 parts for the G1's (module docstring)."""
  core.X2_XML = G1_XML
  core.ROBOT = G1.body
  core.RetargetExtractor = G1RetargetExtractor
  core.ReferenceBuilder = G1ReferenceBuilder
  core.Modes = G1Modes
  cli.reference_joint_map = g1_joint_map
  cli.joint_report = g1_joint_report


if __name__ == "__main__":
  install()
  print(f"[moxi_g1] G1 reference stream: model {G1_XML.name}, robot {G1.body!r}", flush=True)
  sys.exit(cli.main())
