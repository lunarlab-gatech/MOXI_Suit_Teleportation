# MOXI Suit Teleoperation

Teleoperate humanoid robots with the J-MEX MOXI V100R+ mocap suit. Supported now: the
Unitree G1 in MuJoCo, either **kinematic** (your joint angles copied onto the model) or
through the [MorphMimic](https://github.com/lzyang2000/morphmimic) tracking **policy**
with physics.

## Setup

```bash
git clone --recursive <this repo> Mocap_Suit_V100R && cd Mocap_Suit_V100R
git clone https://github.com/unitreerobotics/unitree_mujoco.git ~/unitree_mujoco
curl -LsSf https://astral.sh/uv/install.sh | sh                      # uv, no conda needed
uv sync                                                              # kinematic viewer
(cd tracking/morphmimic && uv sync --group export --group teleop)    # policy (~8 GB)
```

From [Dropbox](https://www.dropbox.com/home/AE-LUNAR/j-mex), put
`MOXI_Player_for_Robot_v0.11.0_20261030/`, `jmex_x2_bridge-0.1.0rc1/` and `license.md`
in `vendor/`, and unzip the checkpoints:

```bash
unzip -q supermimic_ckpts.zip -x '__MACOSX/*' -d tracking/morphmimic/ckpts/supermimic_ckpts
```

Export the policy and the G1 sim scene once:

```bash
cd tracking/morphmimic
uv run --group export python -m deploy.export_onnx \
  ckpts/supermimic_ckpts/supermimic_ckpts/shared_policy_v2 ../../moxi_g1/physics/policy/shared_policy_v2
uv run python ../../moxi_g1/physics/make_stand_clip.py
uv run python -m deploy.sim2sim.export_assets --robot g1 \
  --subset-yaml ../../moxi_g1/physics/data/stand_subset.yaml --clips moxi_stand --out ../../moxi_g1/physics/assets
```

## Run

**MOXI Player.** Connect the PC and the Transponder to a 2.4 GHz phone hotspot, then:

```bash
cd vendor/MOXI_Player_for_Robot_v0.11.0_20261030/MOXI_Player_for_Robot_v0.11.0_20261030
./MOXI_Player_for_Robot_v0.11.0_20261030.x86_64
```

In the Player: `Load Avatar..` → `g1_three_fingers` → `WIFI`, channel `4` → `Setup` →
`Confirm` → `Connect` → stand straight → `Calibrate`.

Then run either the kinematic viewer or the policy, not both.

### Kinematic

```bash
uv run moxi_g1/kinematic/moxi_g1_mujoco.py --channel 4    # --upper-body if you wear only the top
```

Type `c` + Enter to recalibrate, `q` + Enter to quit.

### Policy (sim2sim)

Wear the full suit. In `tracking/morphmimic/`, start the policy and sim first and wait
for `teleop, standing`:

```bash
uv run --group export python ../../vendor/jmex_x2_bridge-0.1.0rc1/jmex_x2_bridge/src/jmex_x2_bridge/tools/gt_live_view.py \
  --deploy ../../moxi_g1/physics/policy/shared_policy_v2 --assets ../../moxi_g1/physics/assets \
  --robot g1 --teleop --interface motion --streamer tcp://127.0.0.1:5556 --viewer native
```

Then the suit stream, in a second terminal:

```bash
uv run --group export --group teleop python ../../moxi_g1/physics/moxi_g1_streamer.py --channel 4 --wait-subscriber
```

Press `c` to calibrate (stand straight), `space` to start or stop following, `x` for off.
`--record run1.npz` saves the session, and `--replay run1.npz` plays it back without the suit.
