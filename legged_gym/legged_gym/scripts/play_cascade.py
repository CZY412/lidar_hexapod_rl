"""el4090_cascade 键盘 play：SE2 步态 + EA2 点云→包络感知全链路演示。

结构来源：

* 键盘后端 / 终端守卫 / 跟随相机 —— 与 ``play_keyboard.py`` 同一套实现
  （pynput 全局监听优先，stdin 超时兜底）；
* 包络可视化 —— 复用 SE2 env 自带的 ``_draw_envelope_condition_vis``
  （``cfg.commands.envelope_debug_viz`` 开启），在其节流包装器内追加
  EA2 187 点云（命中点画红球，无命中不画）；
* 状态行 —— ``params5`` / LiDAR 刷新计数 / stale env 数 / EA2 输出有限性。

依赖：``logs/<SE2 experiment>/model_*.pt``（envelop_2 步态大模型）；
EA2 感知权重已 pin 在 ``envelope_cascade_83/checkpoints/``。
"""

import isaacgym  # noqa: F401  -- Isaac Gym 必须先于 legged_gym 导入

import os
import re
import select
import sys
import termios
import threading
import time
import tty
from collections import deque

import numpy as np
import torch

import legged_gym.envs.el_4090.envelope_cascade_83  # noqa: F401,E402 -- 注册 el4090_cascade
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.el_4090.envelope_cascade_83.ea2_perception import yaw_quat
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.math_utils import quat_apply, quat_apply_yaw

TASK_NAME = "el4090_cascade"

# ── 按键映射（与 play_keyboard.py 一致）────────────────────────────────
HELD_ACTIONS = {
    'w': 'forward',
    's': 'back',
    'a': 'left',
    'd': 'right',
    'q': 'yaw_left',
    'e': 'yaw_right',
}
TAP_ACTIONS = {
    'x': 'stop',
    ' ': 'stop',
    '1': 'gear_0',
    '2': 'gear_1',
    '3': 'gear_2',
    'c': 'camera',
    'v': 'debug_viz',
}
BASE_LIN_VEL_X = 1.0
BASE_LIN_VEL_Y = 0.6
BASE_ANG_VEL_YAW = 0.8
GEARS = (0.5, 1.0, 1.5)
HOLD_TIMEOUT_S = 0.12
DEBUG_VIZ_PERIOD = 5
_ANSI_RE = re.compile(rb'\x1b\[[0-9;]*[A-Za-z]')

GEOMETRY_NAMES = ("front_width", "middle_width", "back_width", "forward_limit", "backward_limit")


# ── 终端 / 键盘后端（与 play_keyboard.py 相同实现）──────────────────────
def _drain_stdin(fd):
    chunks = []
    while select.select([fd], [], [], 0.0)[0]:
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)
    return b''.join(chunks)


class TerminalGuard:
    def __init__(self):
        self.fd = None
        self.old_attrs = None

    def __enter__(self):
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.old_attrs = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            attrs = termios.tcgetattr(self.fd)
            attrs[3] = attrs[3] & ~termios.ECHO
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fd is None:
            return
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_attrs)
        termios.tcflush(self.fd, termios.TCIFLUSH)


class PynputKeyboard:
    def __init__(self):
        from pynput import keyboard as pynput_keyboard
        self._keyboard = pynput_keyboard
        self._lock = threading.Lock()
        self.held = set()
        self.taps = deque()
        self.quit = False

        def canonical(key):
            char = getattr(key, 'char', None)
            if char is not None:
                return char.lower()
            if key == pynput_keyboard.Key.esc:
                return 'esc'
            if key == pynput_keyboard.Key.space:
                return ' '
            return None

        def on_press(key):
            name = canonical(key)
            if name is None:
                return
            if name == 'esc':
                self.quit = True
            elif name in HELD_ACTIONS:
                with self._lock:
                    self.held.add(HELD_ACTIONS[name])
            elif name in TAP_ACTIONS:
                with self._lock:
                    self.taps.append(TAP_ACTIONS[name])

        def on_release(key):
            name = canonical(key)
            if name in HELD_ACTIONS:
                with self._lock:
                    self.held.discard(HELD_ACTIONS[name])

        self._listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    def read(self):
        with self._lock:
            held = set(self.held)
            taps = list(self.taps)
            self.taps.clear()
        return held, taps

    def stop(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


class StdinKeyboard:
    def __init__(self, fd, hold_timeout=HOLD_TIMEOUT_S):
        self.fd = fd
        self.hold_timeout = hold_timeout
        self._last_seen = {}
        self.quit = False

    def read(self):
        now = time.monotonic()
        taps = []
        if self.fd is not None:
            data = _drain_stdin(self.fd)
            data = _ANSI_RE.sub(b'', data)
            for char in data.decode(errors='ignore').lower():
                if char == '\x1b':
                    self.quit = True
                elif char in HELD_ACTIONS:
                    self._last_seen[HELD_ACTIONS[char]] = now
                elif char in TAP_ACTIONS:
                    taps.append(TAP_ACTIONS[char])
        held = {action for action, seen in self._last_seen.items()
                if now - seen <= self.hold_timeout}
        return held, taps

    def stop(self):
        pass


class NullKeyboard:
    """无交互输入（headless / 非 TTY 且 pynput 不可用）时的空后端。"""

    quit = False

    def read(self):
        return set(), []

    def stop(self):
        pass


def make_keyboard(stdin_fd):
    try:
        return PynputKeyboard(), 'pynput（全局监听，真实 keydown/keyup）'
    except Exception as exc:
        if stdin_fd is None:
            print(f"\n[无 TTY 且 pynput 不可用: {exc}] 使用空键盘后端（仅运行，无交互）\n")
            return NullKeyboard(), 'null（无交互输入）'
        print(f"\n[pynput 不可用: {exc}]\n回退到 stdin 模式（{HOLD_TIMEOUT_S:.2f}s 超时判定松手）\n")
        return StdinKeyboard(stdin_fd), 'stdin（兜底）'


def apply_commands(env, cmd):
    env.commands[:, 0] = cmd['x']
    env.commands[:, 1] = cmd['y']
    env.commands[:, 2] = cmd['yaw']  # heading_command=False → 列 2 即 yaw 角速度


def refresh_obs(env):
    if hasattr(env, 'compute_observations'):
        env.compute_observations()
    return env.get_observations()


def follow_camera(env):
    robot_pos = env.root_states[0, :3]
    behind = quat_apply_yaw(env.base_quat[0:1], torch.tensor([[-3.0, 0.0, 2.0]], device=env.device))
    forward = quat_apply_yaw(env.base_quat[0:1], torch.tensor([[1.0, 0.0, 0.0]], device=env.device))
    env.set_camera((robot_pos + behind[0]).cpu().numpy(), (robot_pos + forward[0]).cpu().numpy())


# ── cascade 可视化与状态 ────────────────────────────────────────────────
class CascadeViz:
    """包装 env._draw_debug_vis：节流 + 追加 EA2 点云（命中=红球）。

    env 自带的 _draw_debug_vis 每步开头 clear_lines，因此这里追加的
    点云只存活一帧，与包络六边形的刷新节奏一致。
    """

    MODES = ("每帧", f"每{DEBUG_VIZ_PERIOD}帧", "关闭")

    def __init__(self, env):
        self.env = env
        self.mode = 0
        self.counter = 0
        self._original = env._draw_debug_vis
        env._draw_debug_vis = self._wrapped
        self._red = None

    def _wrapped(self):
        env = self.env
        if self.mode == 2:
            if getattr(env, 'viewer', None) is not None:
                env.gym.clear_lines(env.viewer)
            return
        if self.mode == 1 and self.counter % DEBUG_VIZ_PERIOD != 0:
            return
        self._original()
        self._draw_cascade_cloud()

    def _draw_cascade_cloud(self):
        from isaacgym import gymapi, gymutil

        env = self.env
        cloud = env.cascade_debug_cloud()
        if cloud is None:
            return
        points_body, dists = cloud
        far_plane = float(env.cfg.ea2.far_plane)
        stride = 2
        pts = points_body[0][::stride]
        hit = (dists[0][::stride] < far_plane - 1e-3).cpu().numpy()
        pose_q = env.base_quat[0:1]
        if bool(getattr(env.cfg.ea2, 'yaw_only', False)):
            pose_q = yaw_quat(pose_q)
        n = pts.shape[0]
        world = env.base_pos[0:1] + quat_apply(pose_q.expand(n, 4), pts)
        world = world.detach().cpu().numpy()
        if self._red is None:
            self._red = gymutil.WireframeSphereGeometry(0.04, 4, 4, None, color=(1.0, 0.0, 0.0))
        for pt in world[hit]:
            pose = gymapi.Transform(gymapi.Vec3(float(pt[0]), float(pt[1]), float(pt[2])))
            gymutil.draw_lines(self._red, env.gym, env.viewer, env.envs[0], pose)

    def cycle(self):
        self.mode = (self.mode + 1) % 3
        if self.mode == 2 and getattr(self.env, 'viewer', None) is not None:
            self.env.gym.clear_lines(self.env.viewer)
        return self.MODES[self.mode]

    def tick(self):
        self.counter += 1


def print_cascade_state(env, label=""):
    names = list(env.condition_names)
    condition = env._get_structure_condition()[0].detach().cpu()
    geometry = "  ".join(f"{n}={condition[names.index(n)]:.3f}" for n in GEOMETRY_NAMES)
    priors = "  ".join(
        f"{names[i].replace('morphology_', '').replace('_prior', '')}={condition[i]:.3f}"
        for i in range(5, 8)
    )
    print(f"\n[EA2 包络{label}] {geometry}\n  priors: {priors}")


# ── 主流程 ──────────────────────────────────────────────────────────────
def play(args):
    if args.task != TASK_NAME:
        print(f"play_cascade.py 使用任务 {TASK_NAME!r}（忽略传入的 {args.task!r}）")
        args.task = TASK_NAME

    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)

    # 键盘演示覆盖（与 play_keyboard.py 相同的基线）
    env_cfg.env.num_envs = 1
    env_cfg.env.max_episode_length = 99999
    env_cfg.env.episode_length_s = 99999.0
    env_cfg.commands.resampling_time = 99999
    env_cfg.commands.curriculum = False
    env_cfg.init_state.randomize_rot = True
    env_cfg.init_state.rot_randomization_range = [1.5708, 1.5708]
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.terrain_length = 16.0
    env_cfg.terrain.terrain_width = 16.0
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.terrain_proportions = [0.0] * 7 + [0.5, 0.5]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    # cascade 特有：关感知噪声（训练为 True）、开包络可视化
    env_cfg.ea2.enable_sensor_noise = False
    env_cfg.commands.envelope_debug_viz = True

    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    # 构造期初始姿态与地形穿插（上游既有行为）：做一次正规 reset，
    # 让机器人落到条件预设的出生位姿，再进入演示循环。
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.compute_observations()
    obs = env.get_observations()
    print_cascade_state(env, "（出生默认，EA2 首帧后接管）")
    print(f"\nSE2 obs: {obs.shape[-1]} dims | EA2 obs: 190 dims | "
          f"感知: 187 通道 @ {env_cfg.ea2.update_frequency_hz:.0f}Hz, "
          f"噪声={'开' if env_cfg.ea2.enable_sensor_noise else '关'}")

    se2_ckpt = str(env_cfg.se2_policy.checkpoint).format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    if not os.path.exists(se2_ckpt):
        raise FileNotFoundError(
            f"SE2 步态策略未找到: {se2_ckpt}（见 envelope_cascade_83/checkpoints/README.md）"
        )
    policy = torch.jit.load(se2_ckpt, map_location=env.device)
    policy.eval()
    print(f"SE2 步态策略已加载（TorchScript）: {os.path.basename(se2_ckpt)}")

    camera_on = CAMERA_FOLLOW
    cmd = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
    gear_index = 1

    with TerminalGuard() as term:
        keyboard, backend_name = make_keyboard(term.fd)
        viz = CascadeViz(env) if getattr(env, 'viewer', None) is not None else None
        viz_text = f"v=可视化({'/'.join(CascadeViz.MODES)})  " if viz else ""
        print(f"\n键盘后端: {backend_name}")
        print("w/s=前后  a/d=左右  q/e=转向  x/空格=急停  1/2/3=速度档位  "
              f"{'c=跟随视角  ' if CAMERA_FOLLOW else ''}{viz_text}ESC=退出\n")

        last_print = time.time()
        loop_dt_window = []
        last_cmd_key = None
        step_index = 0
        completed_steps = 0
        try:
            while not keyboard.quit and (
                args.max_steps is None or step_index < args.max_steps
            ):
                step_start = time.time()

                held, taps = keyboard.read()
                for tap in taps:
                    if tap == 'stop':
                        cmd.update(x=0.0, y=0.0, yaw=0.0)
                        held.clear()
                    elif tap == 'camera' and CAMERA_FOLLOW:
                        camera_on = not camera_on
                    elif tap == 'debug_viz' and viz is not None:
                        print(f"\n可视化: {viz.cycle()}")
                    elif tap.startswith('gear_'):
                        gear_index = int(tap.split('_')[1])

                if held:
                    gear = GEARS[gear_index]
                    cmd['x'] = (('forward' in held) - ('back' in held)) * BASE_LIN_VEL_X * gear
                    cmd['y'] = (('left' in held) - ('right' in held)) * BASE_LIN_VEL_Y * gear
                    cmd['yaw'] = (('yaw_left' in held) - ('yaw_right' in held)) * BASE_ANG_VEL_YAW * gear
                else:
                    cmd['x'] = cmd['y'] = cmd['yaw'] = 0.0

                apply_commands(env, cmd)
                cmd_key = (cmd['x'], cmd['y'], cmd['yaw'])
                if cmd_key != last_cmd_key:
                    obs = refresh_obs(env)
                    last_cmd_key = cmd_key

                with torch.no_grad():
                    actions = policy(obs.detach())
                obs, _, rews, dones, infos = env.step(actions.detach())

                if torch.any(dones):
                    apply_commands(env, cmd)
                    obs = refresh_obs(env)
                    last_cmd_key = cmd_key
                    print_cascade_state(env, "（reset 后）")

                if viz is not None:
                    viz.tick()
                if camera_on and step_index % 2 == 0 and getattr(env, 'viewer', None) is not None:
                    follow_camera(env)
                step_index += 1
                completed_steps = step_index

                now = time.time()
                loop_dt_window.append(now - step_start)
                if len(loop_dt_window) > 50:
                    loop_dt_window.pop(0)
                if now - last_print > 0.5:
                    avg = float(np.mean(loop_dt_window))
                    over = "  掉帧!" if avg > env.dt else ""
                    summary = env.cascade_debug_summary()
                    p5 = summary["params5"]
                    print(
                        f"cmd vx={cmd['x']:+.2f} vy={cmd['y']:+.2f} wy={cmd['yaw']:+.2f} | "
                        f"包络[{p5[0]:.2f} {p5[1]:.2f} {p5[2]:.2f} {p5[3]:.2f} {p5[4]:.2f}] | "
                        f"扫描{summary['refresh_count']} stale{summary['stale_envs']} | "
                        f"档位x{GEARS[gear_index]:.1f} | {avg * 1000:.1f}ms/{env.dt * 1000:.0f}ms{over}",
                        end='\r',
                    )
                    last_print = now

                if REALTIME_MODE:
                    sleep_time = env.dt - (time.time() - step_start)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n已退出。")
        finally:
            keyboard.stop()
            print(f"已完成步数: {completed_steps}")


if __name__ == '__main__':
    args = get_args([
        {"name": "--max_steps", "type": int, "default": None,
         "help": "Stop after this many simulation steps (default: interactive)."},
    ])
    REALTIME_MODE = not args.headless
    CAMERA_FOLLOW = True
    play(args)
