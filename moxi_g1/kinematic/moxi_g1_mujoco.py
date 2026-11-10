#!/usr/bin/env python3
"""MOXI Player for Robot (G1 model) -> Unitree G1 in MuJoCo, kinematics only.

MOXI Player retargets the suit wearer onto the G1 and streams its 29 joint angles and
the pelvis pose. This script receives that stream with J-MEX's receiver SDK and writes
it straight into the qpos of unitree_mujoco's g1_29dof model. No physics and no policy:
the robot shows exactly what MOXI Player sends.

From the repo root (the root uv env):

  uv run moxi_g1/kinematic/moxi_g1_mujoco.py                     # live, MOXI Player on this machine
  uv run moxi_g1/kinematic/moxi_g1_mujoco.py --fixed-root        # pelvis held still: joints only
  uv run moxi_g1/kinematic/moxi_g1_mujoco.py --upper-body        # top only: legs held standing
  uv run moxi_g1/kinematic/moxi_g1_mujoco.py --record take1.npz  # also save the qpos trajectory
  uv run moxi_g1/kinematic/moxi_g1_mujoco.py --replay take1.npz  # play a saved take, no suit needed

Type in this terminal, then Enter:
  c  recalibrate: robot back to the origin, facing +x, feet on the floor
  g  toggle ground clamp (lowest foot pinned to the floor every frame)
  f  toggle fixed root (pelvis held at the stand pose)
  q  quit

MOXI's G1 URDF and unitree's g1_29dof.xml share joint names, axes, signs and limits
(checked by FK: legs and arms match to 0.0 mm; the waist roll/pitch axes sit ~1 cm
apart, so the upper body can be off by 1-2 cm when bent at the waist).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO = Path(__file__).resolve().parents[2]  # moxi_g1/kinematic/ -> the repo root
DEFAULT_SDK = REPO / "vendor" / "jmex_x2_bridge-0.1.0rc1" / "MOXIReceiverSDK-python"
DEFAULT_XML = Path.home() / "unitree_mujoco" / "unitree_robots" / "g1" / "scene_29dof.xml"
THIS_HOST = "127.0.0.1"
ROOT_JOINT = 0  # MOXIUrdfJointType.ROOT_JOINT
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
VIEWER_HZ = 60
STATUS_S = 1.0


def joint_key(name: str) -> str:
    """'left_knee_joint', 'left_knee_link', 'Left_Knee' -> 'left_knee'."""
    key = name.strip().lower()
    for suffix in ("_joint", "_link"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key


def yaw_of(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def quat_z(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def quat_mat(q: np.ndarray) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, dtype=np.float64))
    return out.reshape(3, 3)


# --------------------------------------------------------------------------------------
# MOXI side
# --------------------------------------------------------------------------------------


@dataclass
class JointTable:
    names: list[str]
    types: np.ndarray  # MOXIUrdfJointType per joint, -1 if MOXI sent none
    init_rot: np.ndarray  # rad

    @property
    def root(self) -> int | None:
        roots = np.flatnonzero(self.types == ROOT_JOINT)
        return int(roots[0]) if len(roots) else None


@dataclass
class Frame:
    seq: int
    angle: np.ndarray  # [J] rad, MOXI's retargeted joint angle (rotateAngle)
    body_pos: np.ndarray  # [J, 3] MOXI's own FK: global position of each joint's child body
    body_quat: np.ndarray  # [J, 4] wxyz, same chain (root kept upright, yaw only)
    root_quat: np.ndarray  # [4] wxyz, the root as MOXI tracks it, tilt included


class _QuietSdk:
    """stdout that drops whatever the SDK's own code prints (it prints every frame)."""

    def __init__(self, out, prefix: str) -> None:
        self.out, self.prefix = out, prefix

    def write(self, s: str) -> int:
        f = sys._getframe()
        while f is not None:
            if f.f_code.co_filename.startswith(self.prefix):
                return len(s)
            f = f.f_back
        return self.out.write(s)

    def __getattr__(self, name):
        return getattr(self.out, name)


class MoxiSource:
    """Frames from MOXI Player through J-MEX's robot receiver SDK."""

    def __init__(self, sdk_dir: Path, **cfg) -> None:
        if not (sdk_dir / "RobotDemo.py").exists():
            sys.exit(f"MOXI receiver SDK not found in {sdk_dir} (use --sdk)")
        sys.path.insert(0, str(sdk_dir))
        import MOXIReceiver
        import MOXIRobotReceiver
        import RobotDemo
        from MOXIReceiverCfg import MOXIReceiverCfg

        self._sdk, self._rsdk = MOXIReceiver, MOXIRobotReceiver
        self.cfg = MOXIReceiverCfg(**cfg)
        self.robot = RobotDemo.RobotData(self.cfg.channel)
        self.rx = RobotDemo.MxRobotReceiver(self.robot, self.cfg)
        self._quiet = _QuietSdk(sys.stdout, str(sdk_dir) + os.sep)
        sys.stdout = self._quiet  # before the SDK starts its threads
        self.rx.StartSystem()
        self.table: JointTable | None = None
        self._seq: int | None = None
        self.lost = 0

    def poll(self) -> Frame | None:
        self.rx.UpdateSystem()
        r = self.robot
        if not r.finishSetting:  # not connected yet, or disconnected
            self.table, self._seq = None, None
            return None
        if self.table is None:
            idx = sorted(r.moxiJointDict)
            info = [r.GetJointInfoByIndex(i) for i in idx]  # (ok, type, initial rotation)
            self.table = JointTable(
                [r.moxiJointDict[i] for i in idx],
                np.array([t if ok else -1 for ok, t, _ in info]),
                np.array([x if ok else 0.0 for ok, _, x in info], dtype=np.float64),
            )
        if r.seqid < 0 or r.seqid == self._seq:
            return None
        if self._seq is not None and r.seqid - self._seq > 1:
            self.lost += r.seqid - self._seq - 1
        self._seq = r.seqid
        n = len(self.table.names)
        poses = [r.robotPoses[i] for i in range(n)]
        body_quat = np.array([[p.quat.w, p.quat.x, p.quat.y, p.quat.z] for p in poses])
        root = self.table.root
        root_quat = body_quat[root].copy() if root is not None else np.array([1.0, 0, 0, 0])
        if root is not None:  # tracked root rotation (tilt included) from the mocap chain
            q, _ = self._sdk.MxRGetBoneJointMotionData(
                self.cfg.channel, root, self._sdk.MOXI_DATA_TYPE_GLOBAL
            )
            if q is not None:
                ros = [0.0] * 4
                self._rsdk.MxMOXIRotationToRobot(q, ros)
                root_quat = np.array(ros, dtype=np.float64)
        return Frame(
            seq=int(r.seqid),
            angle=np.array([r.rotateAngle[i] for i in range(n)], dtype=np.float64),
            body_pos=np.array([[p.pos.x, p.pos.y, p.pos.z] for p in poses]),
            body_quat=body_quat,
            root_quat=root_quat,
        )

    def close(self) -> None:
        try:
            self.rx.Finish()
        finally:
            if sys.stdout is self._quiet:
                sys.stdout = self._quiet.out


# --------------------------------------------------------------------------------------
# MuJoCo side
# --------------------------------------------------------------------------------------


class G1Kinematic:
    """MOXI frames -> G1 qpos: hinge angles by joint name, pelvis from MOXI's root."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, table: JointTable,
                 subtract_initial: bool = False, upper_body: bool = False) -> None:
        m, d = model, data
        self.m, self.d, self.table = m, d, table
        hinges = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        by_key = {joint_key(m.joint(j).name): j for j in hinges}
        self.root = table.root
        pairs = [(i, by_key[joint_key(n)]) for i, n in enumerate(table.names)
                 if i != self.root and joint_key(n) in by_key]
        self.moxi_idx = np.array([i for i, _ in pairs], dtype=int)
        jnt = np.array([j for _, j in pairs], dtype=int)
        self.joint_names = [m.joint(j).name for j in jnt]
        self.qadr = m.jnt_qposadr[jnt]
        self.lo, self.hi = m.jnt_range[jnt].T
        self.child_body = m.jnt_bodyid[jnt]
        # upper body only: legs held at the stand pose (0), whatever MOXI sends for them
        self.held = np.array([upper_body and any(k in n for k in ("hip", "knee", "ankle"))
                              for n in self.joint_names], dtype=bool)
        self.init_rot = table.init_rot[self.moxi_idx] if subtract_initial else 0.0
        matched = set(self.moxi_idx.tolist())
        self.unmatched = [n for i, n in enumerate(table.names) if i != self.root and i not in matched]
        self.uncovered = [m.joint(j).name for j in hinges if j not in set(jnt.tolist())]

        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.fadr = int(m.jnt_qposadr[free[0]])
        self.pelvis = int(m.jnt_bodyid[free[0]])
        self.home = m.qpos0[self.fadr:self.fadr + 7].copy()
        self.feet = [m.body(n).id for n in FOOT_BODIES]
        d.qpos[:] = m.qpos0
        mujoco.mj_kinematics(m, d)
        self.foot_z0 = float(min(d.xpos[f][2] for f in self.feet))  # ankle height, standing

        # calibration: MOXI root xy and yaw that map to the origin facing +x; z offset
        self.calib: tuple[np.ndarray, float] | None = None
        self.z_off = 0.0
        self.calib_pending = True  # the first frame always sets the height offset

    def report(self) -> str:
        rows = [f"MOXI root joint: {self.table.names[self.root] if self.root is not None else '(none)'}",
                f"driven G1 joints: {len(self.joint_names)}/29"]
        if self.uncovered:
            rows.append(f"  G1 joints MOXI does not drive (held at 0): {', '.join(self.uncovered)}")
        if self.unmatched:
            rows.append(f"  MOXI joints not in this model (ignored): {', '.join(self.unmatched)}")
        return "\n".join(rows)

    def apply(self, fr: Frame, fixed_root: bool, ground_clamp: bool, recalibrate: bool) -> np.ndarray:
        """Write one frame into qpos and run FK. Returns which driven joints were clipped."""
        m, d, a = self.m, self.d, self.fadr
        q = fr.angle[self.moxi_idx] - self.init_rot
        q[self.held] = 0.0
        clipped = (q < self.lo) | (q > self.hi)
        d.qpos[self.qadr] = np.clip(q, self.lo, self.hi)
        if fixed_root or self.root is None:
            d.qpos[a:a + 7] = self.home
            mujoco.mj_kinematics(m, d)
            return clipped

        p, rq = fr.body_pos[self.root], fr.root_quat / np.linalg.norm(fr.root_quat)
        if self.calib is None or recalibrate:
            self.calib = (np.array([p[0], p[1], 0.0]), yaw_of(rq))
        p0, yaw0 = self.calib
        d.qpos[a:a + 3] = quat_mat(quat_z(-yaw0)) @ (p - p0)
        d.qpos[a + 3:a + 7] = quat_mul(quat_z(-yaw0), rq)
        mujoco.mj_kinematics(m, d)
        if recalibrate or ground_clamp or self.calib_pending:
            self.z_off = self.foot_z0 - float(min(d.xpos[f][2] for f in self.feet))
            self.calib_pending = False
        d.qpos[a + 2] += self.z_off
        mujoco.mj_kinematics(m, d)
        return clipped

    def fk_error(self, fr: Frame) -> np.ndarray | None:
        """Per driven joint: distance between MOXI's own FK and MuJoCo's, in the root frame.
        A few mm to ~2 cm is normal; 10+ cm means a sign/axis/zero mismatch."""
        if self.root is None:
            return None
        d = self.d
        r_moxi = quat_mat(fr.body_quat[self.root])
        moxi = (fr.body_pos[self.moxi_idx] - fr.body_pos[self.root]) @ r_moxi
        r_pel = d.xmat[self.pelvis].reshape(3, 3)
        ours = (d.xpos[self.child_body] - d.xpos[self.pelvis]) @ r_pel
        err = np.linalg.norm(moxi - ours, axis=1)
        err[self.held] = 0.0  # held legs differ from MOXI by design, not a mapping error
        return err


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def _start_stdin_reader(cmds: "queue.Queue[str]") -> None:
    def run():
        try:
            for line in sys.stdin:
                s = line.strip().lower()
                if s:
                    cmds.put(s[0])
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()


def _close_viewer(viewer) -> None:
    viewer.close()
    deadline = time.monotonic() + 2.0
    while viewer.is_running() and time.monotonic() < deadline:
        time.sleep(0.01)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--xml", type=Path, default=DEFAULT_XML, help=f"G1 scene (default {DEFAULT_XML})")
    p.add_argument("--sdk", type=Path, default=DEFAULT_SDK, help="MOXIReceiverSDK-python folder")
    p.add_argument("--replay", type=Path, help="play a take saved with --record instead of MOXI")
    p.add_argument("--record", type=Path, help="save the qpos trajectory here (.npz) on exit")
    p.add_argument("--fixed-root", action="store_true", help="hold the pelvis at the stand pose")
    p.add_argument("--upper-body", action="store_true",
                   help="top only: pelvis fixed and legs held standing; waist and arms follow MOXI")
    p.add_argument("--ground-clamp", action="store_true",
                   help="pin the lowest foot to the floor every frame (no drift, no jumps)")
    p.add_argument("--subtract-initial", action="store_true",
                   help="hinge angle = rotateAngle - initialRotation (try if the zero pose is off)")
    p.add_argument("--no-viewer", action="store_true", help="print status only")
    p.add_argument("--channel", type=int, default=1, help="MOXI channel (default 1)")
    p.add_argument("--tcp-ip", default=THIS_HOST,
                   help=f"this host's address MOXI Player connects to (default {THIS_HOST})")
    p.add_argument("--tcp-port", type=int, default=7000)
    p.add_argument("--udp-broadcast", default=THIS_HOST,
                   help=f"where the SDK announces itself (default {THIS_HOST}: same machine; "
                        "255.255.255.255 for a MOXI Player elsewhere on the LAN)")
    p.add_argument("--udp-port", type=int, default=10100)
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_kinematics(model, data)

    viewer = None
    if not args.no_viewer:
        viewer =mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = int(model.jnt_bodyid[free[0]])
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 180.0, -15.0, 3.0  # facing the robot
    lock = viewer.lock if viewer is not None else contextlib.nullcontext

    cmds: queue.Queue[str] = queue.Queue()
    _start_stdin_reader(cmds)
    fixed_root, ground_clamp = args.fixed_root or args.upper_body, args.ground_clamp

    source = None
    replay = None
    if args.replay:
        z = np.load(args.replay)
        replay = (z["t"] - z["t"][0], z["qpos"])
        print(f"[g1] replaying {len(replay[0])} frames ({replay[0][-1]:.1f} s) from {args.replay}, looping")
    else:
        source = MoxiSource(args.sdk, channel=args.channel, tcp_server_ip=args.tcp_ip,
                            tcp_server_port=args.tcp_port, udp_broadcast_ip=args.udp_broadcast,
                            udp_sender_start_port=args.udp_port)
        print(f"[g1] waiting for MOXI Player on channel {args.channel} "
              f"(tcp {args.tcp_ip}:{args.tcp_port}, udp {args.udp_broadcast}:{args.udp_port}) ...")
    print("[g1] commands: c=recalibrate  g=ground clamp  f=fixed root  q=quit  (then Enter)", flush=True)

    g1: G1Kinematic | None = None
    rec_t: list[float] = []
    rec_q: list[np.ndarray] = []
    n, t_status, t_draw, t0 = 0, time.monotonic(), 0.0, time.monotonic()
    last_clipped = np.zeros(0, dtype=bool)
    last_err = None
    recal = False
    try:
        while viewer is None or viewer.is_running():
            while not cmds.empty():
                c = cmds.get()
                if c == "q":
                    raise KeyboardInterrupt
                if c == "c":
                    recal = True
                    print("[g1] recalibrating on the next frame", flush=True)
                elif c == "g":
                    ground_clamp = not ground_clamp
                    print(f"[g1] ground clamp {'on' if ground_clamp else 'off'}", flush=True)
                elif c == "f":
                    fixed_root = not fixed_root
                    print(f"[g1] fixed root {'on' if fixed_root else 'off'}", flush=True)

            if replay is not None:
                ts, qs = replay
                k = int(np.searchsorted(ts, (time.monotonic() - t0) % max(ts[-1], 1e-6)))
                with lock():
                    data.qpos[:] = qs[min(k, len(qs) - 1)]
                    mujoco.mj_kinematics(model, data)
                n += 1
                time.sleep(1.0 / VIEWER_HZ)
            else:
                frame = source.poll()
                if frame is None:
                    if source.table is None and g1 is not None:
                        print("[g1] MOXI Player disconnected, waiting ...", flush=True)
                        g1 = None
                    time.sleep(0.001)
                    continue
                if g1 is None or g1.table is not source.table:  # first frame or reconnect
                    g1 = G1Kinematic(model, data, source.table, args.subtract_initial, args.upper_body)
                    print("[g1] MOXI Player connected\n" + g1.report(), flush=True)
                with lock():
                    last_clipped = g1.apply(frame, fixed_root, ground_clamp, recal)
                    last_err = g1.fk_error(frame)
                recal = False
                n += 1
                if args.record:
                    rec_t.append(time.monotonic())
                    rec_q.append(data.qpos.copy())

            now = time.monotonic()
            if viewer is not None and now - t_draw >= 1.0 / VIEWER_HZ:
                viewer.sync()
                t_draw = now
            if now - t_status >= STATUS_S:
                msg = f"[g1] {n / (now - t_status):4.0f} fps"
                if source is not None and g1 is not None:
                    msg += f" | lost {source.lost} | pelvis z {data.qpos[g1.fadr + 2]:.2f} m"
                    if last_err is not None and len(last_err):
                        w = int(np.argmax(last_err))
                        msg += (f" | FK err mean {1e3 * last_err.mean():.0f} mm, max "
                                f"{1e3 * last_err[w]:.0f} mm ({g1.joint_names[w]})")
                    if last_clipped.any():
                        msg += " | clipped: " + ", ".join(
                            g1.joint_names[i] for i in np.flatnonzero(last_clipped))
                print(msg, flush=True)
                n, t_status = 0, now
    except KeyboardInterrupt:
        pass
    finally:
        if source is not None:
            source.close()
        if args.record and rec_q:
            np.savez_compressed(args.record, t=np.array(rec_t) - rec_t[0], qpos=np.stack(rec_q),
                                joint_names=np.array([model.joint(j).name for j in range(model.njnt)]),
                                xml=str(args.xml))
            print(f"[g1] saved {len(rec_q)} frames to {args.record}", flush=True)
        if viewer is not None:
            _close_viewer(viewer)
            sys.stdout.flush()
            os._exit(0)  # GLFW teardown after a passive viewer can crash on X11


if __name__ == "__main__":
    main()
