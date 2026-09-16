"""单机器人键盘控制 play 脚本（地图/环境数固定）。

支持两种键盘后端，接口一致（``read() -> (held, taps)``）：

* **pynput（首选）**：X11 全局按键监听，能拿到真实的 keydown / keyup，
  因此"按住即走、松手即停"完全成立，且不存在终端输入队列堆积。
  需要先在运行 isaacgym 的 Python 环境里 ``pip install pynput``。
* **stdin（兜底）**：终端没有 keyup 事件，只能"猜"——某个方向键最后一次
  出现后 ``HOLD_TIMEOUT_S`` 内视为按住。每帧用 ``select(0)`` 把输入队列
  **排空**，避免积压导致的滞后。

两种模式下终端都会被设为 cbreak + 无回显，并在退出时 ``tcflush`` 丢弃输入
队列，所以不会有一大堆字母泄漏回 shell。
"""

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
import re
import select
import termios
import threading
import tty
import time
from collections import deque

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.math_utils import quat_apply_yaw

import numpy as np
import torch


# ── 按键映射 ────────────────────────────────────────────────────────────
# 持续型：按住生效，松手归零
HELD_ACTIONS = {
    'w': 'forward',
    's': 'back',
    'a': 'left',
    'd': 'right',
    'q': 'yaw_left',
    'e': 'yaw_right',
}
# 单次型：按一下触发一次
TAP_ACTIONS = {
    'x': 'stop',
    ' ': 'stop',
    '1': 'gear_0',
    '2': 'gear_1',
    '3': 'gear_2',
    'c': 'camera',
    'v': 'debug_viz',
}

# 基准速度（会被 1/2/3 档位缩放）
BASE_LIN_VEL_X = 1.0
BASE_LIN_VEL_Y = 0.6
BASE_ANG_VEL_YAW = 0.8
GEARS = (0.5, 1.0, 1.5)

HOLD_TIMEOUT_S = 0.12   # 仅 stdin 兜底模式：超过此时间没收到重复字符即视为松手
DEBUG_VIZ_PERIOD = 5    # v 键的中间档：每 N 个控制步画一次包络
_ANSI_RE = re.compile(rb'\x1b\[[0-9;]*[A-Za-z]')


# ── 终端处理 ────────────────────────────────────────────────────────────
def _drain_stdin(fd):
    """把输入队列一次性读完，防止积压导致指令滞后 / 退出时泄漏到 shell。"""
    chunks = []
    while select.select([fd], [], [], 0.0)[0]:
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)
    return b''.join(chunks)


class TerminalGuard:
    """cbreak + 无回显（保留 ISIG，Ctrl+C 仍然可用），退出时冲刷输入队列。"""

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
        termios.tcflush(self.fd, termios.TCIFLUSH)   # 丢弃积压的字符


# ── 键盘后端 ────────────────────────────────────────────────────────────
class PynputKeyboard:
    """全局按键监听：真实的 keydown/keyup，按住即持续，松手立刻归零。"""

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
                return
            if name in HELD_ACTIONS:
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

        self._listener = pynput_keyboard.Listener(on_press=on_press,
                                                  on_release=on_release)
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
    """兜底后端：终端无 keyup，用"最后一次出现时间 + 超时"伪造按住状态。"""

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
            data = _ANSI_RE.sub(b'', data)          # 忽略方向键等转义序列
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


def make_keyboard(stdin_fd):
    try:
        return PynputKeyboard(), 'pynput（全局监听，真实 keydown/keyup）'
    except Exception as exc:      # 未安装 / 无 X server / 权限不足
        if stdin_fd is None:
            raise
        print(f"\n[pynput 不可用: {exc}]\n"
              f"回退到终端 stdin 模式（松手到停止约 {HOLD_TIMEOUT_S:.2f}s）。\n"
              f"如需长按手感：pip install pynput\n")
        return StdinKeyboard(stdin_fd), 'stdin（兜底，无 keyup，靠超时判定松手）'


# ── 指令写入 ────────────────────────────────────────────────────────────
def apply_commands(env, cmd):
    """把键盘指令写进 env.commands。

    heading_command=False 时 commands[:, 2] 是 yaw 角速度（本任务即如此）；
    heading_command=True 时 commands[:, 2] 每帧会被 _recompute_ang_vel 覆盖，
    必须改写成 commands[:, 3] 的 heading 目标角。
    """
    env.commands[:, 0] = cmd['x']
    env.commands[:, 1] = cmd['y']
    if getattr(env.cfg.commands, 'heading_command', False):
        cmd['heading'] += cmd['yaw'] * env.dt
        env.commands[:, 3] = cmd['heading']
        env.commands[:, 2] = 0.0
    else:
        env.commands[:, 2] = cmd['yaw']


def refresh_obs(env):
    """重算观测，让刚写入的指令立刻被策略看到。

    注意：el4090_ea 之类的 env 在 step() 末尾已经自行调用过
    compute_observations()，所以这里只在"指令发生变化"时才补算一次，
    稳态下零开销（见主循环里的 cmd_key 判断）。
    """
    if hasattr(env, 'compute_observations'):
        env.compute_observations()
    return env.get_observations()


class DebugVizThrottle:
    """包络/LiDAR 可视化节流器（v 键切换）。

    el4090_ea 的 _draw_envelope 每步要发 288 次 add_lines（两个六棱柱），
    _draw_lidar_points 还要对每个命中点单独 draw_lines —— 5760 条射线下
    这是最大的掉帧来源。这里在不改动 env 的前提下包一层，支持
    每帧 / 每 DEBUG_VIZ_PERIOD 帧 / 关闭 三档。
    """

    MODES = ("每帧", f"每{DEBUG_VIZ_PERIOD}帧", "关闭")

    def __init__(self, env, period=DEBUG_VIZ_PERIOD):
        self.env = env
        self.period = max(1, period)
        self.mode = 0
        self.counter = 0
        self._wrapped = {}
        for name in ("_draw_envelope", "_draw_velocity_arrows"):
            original = getattr(env, name, None)
            if callable(original):
                self._wrapped[name] = original
        self.install()

    def install(self):
        throttle = self

        def make_wrapper(fn):
            def wrapper(*a, **kw):
                if throttle.mode == 2:                     # 关闭
                    return None
                if throttle.mode == 1 and throttle.counter % throttle.period != 0:
                    return None
                return fn(*a, **kw)
            return wrapper

        for name, original in self._wrapped.items():
            setattr(self.env, name, make_wrapper(original))

    def restore(self):
        for name, original in self._wrapped.items():
            setattr(self.env, name, original)

    def tick(self):
        self.counter += 1

    def cycle(self):
        self.mode = (self.mode + 1) % 3
        if self.mode == 2 and getattr(self.env, 'viewer', None) is not None:
            self.env.gym.clear_lines(self.env.viewer)     # 关掉时清一次残留
        return self.MODES[self.mode]


def follow_camera(env):
    robot_pos = env.root_states[0, :3]
    robot_quat = env.base_quat[0:1]
    behind = quat_apply_yaw(robot_quat, torch.tensor([[-3.0, 0.0, 2.0]],
                                                     device=env.device))
    forward = quat_apply_yaw(robot_quat, torch.tensor([[1.0, 0.0, 0.0]],
                                                      device=env.device))
    env.set_camera((robot_pos + behind[0]).cpu().numpy(),
                   (robot_pos + forward[0]).cpu().numpy())


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # ── Single-robot keyboard-control overrides ──
    env_cfg.env.num_envs = 1
    env_cfg.env.max_episode_length = 99999     # disable timeout reset
    env_cfg.env.episode_length_s = 99999.0
    env_cfg.commands.resampling_time = 99999   # disable automatic command resampling
    env_cfg.commands.curriculum = False
    env_cfg.init_state.randomize_rot = True
    env_cfg.init_state.rot_randomization_range = [1.5708, 1.5708]  # +90° yaw
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.terrain_length = 16.0
    env_cfg.terrain.terrain_width = 16.0
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    camera_on = CAMERA_FOLLOW
    cmd = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'heading': 0.0}
    gear_index = 1

    with TerminalGuard() as term:
        if term.fd is None and not _pynput_available():
            print("stdin 不是 TTY，且 pynput 不可用：键盘控制已禁用。")
            return
        keyboard, backend_name = make_keyboard(term.fd)
        viz = DebugVizThrottle(env) if getattr(env, 'viewer', None) is not None else None

        viz_text = f"v=包络绘制({'/'.join(DebugVizThrottle.MODES)})  " if viz else ""
        print(f"\n键盘后端: {backend_name}")
        print("w/s=前后  a/d=左右  q/e=转向  x/空格=急停  1/2/3=速度档位  "
              f"{'c=切换跟随视角  ' if CAMERA_FOLLOW else ''}{viz_text}ESC=退出\n")

        last_print = time.time()
        loop_dt_window = []
        last_cmd_key = None
        step_index = 0
        try:
            while not keyboard.quit:
                step_start = time.time()

                held, taps = keyboard.read()
                for tap in taps:
                    if tap == 'stop':
                        cmd.update(x=0.0, y=0.0, yaw=0.0)
                        held.clear()
                    elif tap == 'camera' and CAMERA_FOLLOW:
                        camera_on = not camera_on
                    elif tap == 'debug_viz' and viz is not None:
                        print(f"\n包络绘制: {viz.cycle()}")
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
                # 只在指令变化时重算观测：稳态零开销，按键时仍然零延迟
                cmd_key = (cmd['x'], cmd['y'], cmd['yaw'])
                if cmd_key != last_cmd_key:
                    obs = refresh_obs(env)
                    last_cmd_key = cmd_key

                actions = policy(obs.detach())
                obs, _, rews, dones, infos = env.step(actions.detach())

                if torch.any(dones):
                    # reset 会重采样指令并重建观测，写回键盘指令并强制刷新
                    apply_commands(env, cmd)
                    obs = refresh_obs(env)
                    last_cmd_key = cmd_key

                if viz is not None:
                    viz.tick()
                if camera_on and step_index % 2 == 0:
                    follow_camera(env)
                step_index += 1

                now = time.time()
                loop_dt_window.append(now - step_start)
                if len(loop_dt_window) > 50:
                    loop_dt_window.pop(0)
                if now - last_print > 0.25:
                    avg = float(np.mean(loop_dt_window))
                    over = "  掉帧!" if avg > env.dt else ""
                    print(f"cmd: vx={cmd['x']:+.2f} vy={cmd['y']:+.2f} "
                          f"wyaw={cmd['yaw']:+.2f} | 档位x{GEARS[gear_index]:.1f} | "
                          f"单步 {avg * 1000:.1f}ms / 预算 {env.dt * 1000:.1f}ms{over}",
                          end='\r')
                    last_print = now

                if REALTIME_MODE:
                    sleep_time = env.dt - (now - step_start)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n已退出。")
        finally:
            keyboard.stop()
            print()


def _pynput_available():
    try:
        import pynput  # noqa: F401
        return True
    except Exception:
        return False


if __name__ == '__main__':
    REALTIME_MODE = True
    CAMERA_FOLLOW = True  # True=跟随机器人, False=固定视角
    args = get_args()
    play(args)
