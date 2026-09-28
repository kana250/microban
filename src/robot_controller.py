# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud / Adapted for Orange Pi Zero 2W + Microban Custom HAT

import os
import sys
import time
import numpy as np

# Ensure custom hat library is in sys.path
CUSTOM_HAT_SRC = "/home/orangepi/src/khr3-mujoco-control/integrations/microban/custom_hat/src"
if CUSTOM_HAT_SRC not in sys.path:
    sys.path.insert(0, CUSTOM_HAT_SRC)

from microban_custom_hat.config import load_config
from microban_custom_hat.dynamixel import DynamixelBus
from microban_custom_hat.bno055 import ThreadedBNO055Reader

from constants import MOTOR_TO_ID, MOTOR_SIGN, PRESENT_CURRENT_UNIT_A

CONFIG_PATH = "/home/orangepi/src/khr3-mujoco-control/integrations/microban/custom_hat/config/microban_auto_hat_orangepi_zero2w.yaml"

# Motor groups for robust Sync Read & soft-start
SYNC_GROUPS = [
    [21, 22, 23, 24, 25, 26],  # Right leg (6 axes)
    [11, 12, 13, 14, 15, 16],  # Left leg (6 axes)
    [41, 42, 43],              # Right arm (3 axes)
    [31, 32, 33],              # Left arm (3 axes)
    [51],                      # Head (1 axis)
]


class RobotController:
    """Orange Pi Zero 2W + Microban Custom HAT implementation of RobotController.

    Drop-in replacement for the official Rhoban Microban RobotController.
    """

    @property
    def is_closed(self) -> bool:
        return self._closed

    def __init__(self, serial_port: str = None, baudrate: int = None, timeout: float = 0.1) -> None:
        self._closed = False
        self.cfg = load_config(CONFIG_PATH)
        self._id_to_sign: dict[int, float] = {MOTOR_TO_ID[name]: MOTOR_SIGN[name] for name in MOTOR_TO_ID}

        # 1. Open Dynamixel bus via high-precision GPIO bit-bang UART (1 Mbps)
        print("[RobotController] Opening Dynamixel bus on /dev/ttyS5 (1 Mbps)...")
        self._bus = DynamixelBus.open(self.cfg.uart)
        print("[RobotController] Dynamixel bus ready.")

        # 2. Open BNO055 IMU via I2C-2 (100 Hz)
        print("[RobotController] Starting BNO055 IMU reader on I2C-2 (100 Hz)...")
        self._imu_reader = ThreadedBNO055Reader.open(self.cfg.imu)
        self._imu_reader.start()
        print("[RobotController] BNO055 IMU reader running.")

        # Motion cache
        self._pos_cache: dict[int, float] = {}
        self._vel_cache: dict[int, float] = {}
        self._last_read_time: float = 0.0
        self._last_read_ids: tuple[int, ...] = ()
        self._torque_enabled_ids: set[int] = set()

        # Position reads are deliberately deferred to hold_present_position().
        # Reading all 19 servos twice during startup can overrun the bit-bang UART.
        # The cache only provides a defined fallback for non-startup callers.
        print("[RobotController] Initializing empty motion cache...")
        for mid in MOTOR_TO_ID.values():
            self._pos_cache[mid] = 0.0
            self._vel_cache[mid] = 0.0
        print("[RobotController] Motion cache ready; hardware baseline is deferred.")

    def _rad_to_tick(self, rad: float, motor_id: int) -> int:
        hw_rad = rad * self._id_to_sign[motor_id]
        if motor_id in (11, 21):
            # Hip yaw joints (ID 11 left_hip_yaw, ID 21 right_hip_yaw) have physical center at 0/4096 ticks
            t = round(hw_rad * 2048.0 / np.pi)
            tick = t if t >= 0 else t + 4096
            return max(0, min(4095, tick))
        else:
            tick = round(2048 + hw_rad * 2048.0 / np.pi)
            return max(0, min(4095, tick))

    def _tick_to_rad(self, tick: int, motor_id: int) -> float:
        if motor_id in (11, 21):
            # Hip yaw joints (ID 11 left_hip_yaw, ID 21 right_hip_yaw) have physical center at 0/4096 ticks
            t = tick if tick < 2048 else tick - 4096
            hw_rad = t * np.pi / 2048.0
            return hw_rad * self._id_to_sign[motor_id]
        else:
            hw_rad = (tick - 2048) * np.pi / 2048.0
            return hw_rad * self._id_to_sign[motor_id]

    def _raw_vel_to_rad_s(self, raw: int, motor_id: int) -> float:
        val = raw - 0x100000000 if raw >= 0x80000000 else raw
        rad_s = val * 0.229 * np.pi / 30.0
        return rad_s * self._id_to_sign[motor_id]

    def _update_motion(self, ids: list[int]) -> None:
        """Batch-read positions and velocities in groups using sync_read_motion."""
        now = time.perf_counter()
        target_ids = set(ids)

        for grp in SYNC_GROUPS:
            sub = [mid for mid in grp if mid in target_ids]
            if not sub:
                continue
            try:
                motion = self._bus.sync_read_motion(sub)
                for mid, tick, raw_vel in zip(sub, motion.position_ticks, motion.velocity_raw):
                    self._pos_cache[mid] = self._tick_to_rad(tick, mid)
                    self._vel_cache[mid] = self._raw_vel_to_rad_s(raw_vel, mid)
            except Exception:
                try:
                    self._bus._port.clearPort()
                except Exception:
                    pass

        self._last_read_time = now
        self._last_read_ids = tuple(ids)

    def sync_write_torque_enable(self, ids: list[int], values: list[bool]) -> None:
        """Enable or disable torque, rolling back partial soft-start failures."""
        if len(ids) != len(values):
            raise ValueError("torque enable IDs and values must have equal length")
        if not ids:
            raise ValueError("at least one motor ID is required")

        all_false = not any(values)
        if all_false:
            self._bus.sync_write_torque_enable(ids, [False] * len(ids))
            self._torque_enabled_ids.difference_update(ids)
            return
        if not all(values):
            raise ValueError("mixed torque enable values are not supported")

        # Staggered soft-start (legs -> arms -> head) to prevent voltage brown-out
        requested = set(ids)
        grouped_ids = {mid for group in SYNC_GROUPS for mid in group}
        if not requested.issubset(grouped_ids):
            unknown = sorted(requested - grouped_ids)
            raise ValueError(f"motor IDs are missing from SYNC_GROUPS: {unknown}")

        try:
            for grp in SYNC_GROUPS:
                sub_ids = [mid for mid in grp if mid in requested]
                if sub_ids:
                    self._bus.sync_write_torque_enable(sub_ids, [True] * len(sub_ids))
                    self._torque_enabled_ids.update(sub_ids)
                    time.sleep(0.04)  # 40ms stagger between groups
        except Exception:
            # A failed group may leave earlier groups powered. Best-effort rollback
            # must happen before propagating the startup failure.
            try:
                self._bus.sync_write_torque_enable(ids, [False] * len(ids))
            finally:
                self._torque_enabled_ids.difference_update(ids)
            raise

    def hold_present_position(self, ids: list[int]) -> list[float]:
        """Prime Goal Position from exact hardware ticks while torque is disabled.

        Every position read must succeed. A hardware-alert status packet or a
        missing servo therefore aborts startup before any torque is enabled.
        """
        if self._torque_enabled_ids.intersection(ids):
            raise RuntimeError("cannot prime Goal Position while torque is enabled")

        ticks: list[int] = []
        positions: list[float] = []
        for index, motor_id in enumerate(ids):
            tick = self._bus.read_present_position_tick(motor_id)
            ticks.append(tick)
            position = self._tick_to_rad(tick, motor_id)
            positions.append(position)
            self._pos_cache[motor_id] = position
            self._vel_cache[motor_id] = 0.0
            if index + 1 < len(ids):
                time.sleep(0.01)

        self._bus.sync_write_goal_ticks(ids, ticks)
        print("[RobotController] Goal Position primed from current hardware position.")
        return positions

    def sync_write_status_return_level(self, ids: list[int], levels: list[int]) -> None:
        self._bus.sync_write_status_return_level(ids, levels)

    def sync_write_goal_position(self, ids: list[int], positions: list[float]) -> None:
        ticks = [self._rad_to_tick(pos, motor_id) for motor_id, pos in zip(ids, positions)]
        self._bus.sync_write_goal_ticks(ids, ticks)

    def sync_read_present_position(self, ids: list[int]) -> list[float]:
        self._update_motion(ids)
        return [self._pos_cache[mid] for mid in ids]

    def read_present_position(self, motor_id: int) -> float:
        try:
            tick = self._bus.read_present_position_tick(motor_id)
            self._pos_cache[motor_id] = self._tick_to_rad(tick, motor_id)
        except Exception:
            pass
        return self._pos_cache[motor_id]

    def sync_read_present_velocity(self, ids: list[int]) -> list[float]:
        if (time.perf_counter() - self._last_read_time) > 0.005 or self._last_read_ids != tuple(ids):
            self._update_motion(ids)
        return [self._vel_cache[mid] for mid in ids]

    def read_present_velocity(self, motor_id: int) -> float:
        return self._vel_cache.get(motor_id, 0.0)

    def sync_read_present_current(self, ids: list[int]) -> list[float]:
        try:
            raws = self._bus.sync_read_present_current_raw(ids)
            result = []
            for r in raws:
                val = r - 0x10000 if r >= 0x8000 else r
                result.append(float(val) * PRESENT_CURRENT_UNIT_A)
            return result
        except Exception:
            return [0.0] * len(ids)

    def sync_read_present_input_voltage(self, ids: list[int]) -> list[float]:
        try:
            raws = self._bus.sync_read_present_input_voltage_raw(ids)
            return [float(r) * 0.1 for r in raws]
        except Exception:
            return [7.4] * len(ids)

    def read_present_input_voltage(self, motor_id: int) -> float:
        try:
            raw = self._bus.read_present_input_voltage_raw(motor_id)
            return float(raw) * 0.1
        except Exception:
            return 7.4

    def sync_read_kp(self, ids: list[int]) -> list[int]:
        try:
            return [int(v) for v in self._bus.sync_read_position_p_gain(ids)]
        except Exception:
            return [400] * len(ids)

    def sync_write_kp(self, ids: list[int], gains: list[int]) -> None:
        self._bus.sync_write_position_p_gain(ids, gains)

    def read_acc(self) -> tuple[float, float, float]:
        """Return raw accelerometer (ax, ay, az) in g."""
        snap = self._imu_reader.get_latest(require_fresh=False)
        return snap.body_accel_g

    def read_gyro(self) -> tuple[float, float, float]:
        """Return (gx, gy, gz) in rad/s."""
        snap = self._imu_reader.get_latest(require_fresh=False)
        return snap.body_gyro_rad_s

    def read_quat(self, dt: float) -> tuple[float, float, float, float]:
        """Return orientation quaternion (w, x, y, z) in body frame."""
        _ = dt
        snap = self._imu_reader.get_latest(require_fresh=False)
        return snap.body_quat_wxyz

    def get_imu_status(self) -> dict[str, float | int | bool]:
        snap = self._imu_reader.get_latest(require_fresh=False)
        return {
            "valid": snap.valid,
            "error_count": snap.error_count,
        }

    def shutdown(self) -> None:
        if self._closed:
            return

        if self._torque_enabled_ids:
            enabled_ids = sorted(self._torque_enabled_ids)
            try:
                self._bus.sync_write_torque_enable(
                    enabled_ids, [False] * len(enabled_ids)
                )
                self._torque_enabled_ids.clear()
            except Exception as exc:
                print(f"WARNING: emergency torque disable failed: {exc}")
        try:
            self._imu_reader.stop()
        except Exception:
            pass
        try:
            self._bus.close()
        except Exception:
            pass
        self._closed = True

    def close(self) -> None:
        self.shutdown()
