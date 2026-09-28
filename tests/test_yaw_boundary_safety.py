import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_robot_controller_module():
    constants = types.ModuleType("constants")
    constants.MOTOR_TO_ID = {"left_hip_yaw": 11, "right_hip_yaw": 21}
    constants.MOTOR_SIGN = {"left_hip_yaw": -1.0, "right_hip_yaw": -1.0}
    constants.PRESENT_CURRENT_UNIT_A = 0.001

    config = types.ModuleType("microban_custom_hat.config")
    config.load_config = lambda path: None
    dynamixel = types.ModuleType("microban_custom_hat.dynamixel")
    dynamixel.DynamixelBus = object
    bno055 = types.ModuleType("microban_custom_hat.bno055")
    bno055.ThreadedBNO055Reader = object

    modules = {
        "constants": constants,
        "microban_custom_hat": types.ModuleType("microban_custom_hat"),
        "microban_custom_hat.config": config,
        "microban_custom_hat.dynamixel": dynamixel,
        "microban_custom_hat.bno055": bno055,
    }
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            "microban_test_robot_controller", ROOT / "src" / "robot_controller.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class HipYawBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_robot_controller_module()

    def controller(self, edge):
        controller = self.module.RobotController.__new__(self.module.RobotController)
        controller._id_to_sign = {11: -1.0, 21: -1.0}
        controller._hip_yaw_edge = {11: edge}
        return controller

    def test_low_edge_never_wraps_to_4095_side(self):
        controller = self.controller(0)
        self.assertEqual(controller._rad_to_tick(0.1, 11), 0)
        self.assertGreater(controller._rad_to_tick(-0.1, 11), 0)

    def test_high_edge_never_wraps_to_zero_side(self):
        controller = self.controller(4095)
        self.assertEqual(controller._rad_to_tick(-0.1, 11), 4095)
        self.assertLess(controller._rad_to_tick(0.1, 11), 4095)

    def test_zero_command_stays_on_selected_edge(self):
        self.assertEqual(self.controller(0)._rad_to_tick(0.0, 11), 0)
        self.assertEqual(self.controller(4095)._rad_to_tick(0.0, 11), 4095)

    def test_startup_normalizes_positive_multi_turn_position(self):
        controller = self.controller(None)
        controller._hip_yaw_edge = {}
        controller._torque_enabled_ids = set()
        controller._pos_cache = {}
        controller._vel_cache = {}
        controller._bus = FakeBus({11: 4149})

        positions = controller.hold_present_position([11])

        self.assertEqual(controller._bus.goal_write, ([11], [53]))
        self.assertAlmostEqual(positions[0], controller._tick_to_rad(53, 11))

    def test_startup_normalizes_negative_multi_turn_position(self):
        controller = self.controller(None)
        controller._hip_yaw_edge = {}
        controller._torque_enabled_ids = set()
        controller._pos_cache = {}
        controller._vel_cache = {}
        controller._bus = FakeBus({11: -1})

        positions = controller.hold_present_position([11])

        self.assertEqual(controller._bus.goal_write, ([11], [4095]))
        self.assertAlmostEqual(positions[0], controller._tick_to_rad(4095, 11))

    def test_startup_normalizes_multi_turn_position_for_other_joints(self):
        controller = self.controller(None)
        controller._id_to_sign[12] = 1.0
        controller._hip_yaw_edge = {}
        controller._torque_enabled_ids = set()
        controller._pos_cache = {}
        controller._vel_cache = {}
        controller._bus = FakeBus({12: 4096 + 2048})

        positions = controller.hold_present_position([12])

        self.assertEqual(controller._bus.goal_write, ([12], [2048]))
        self.assertAlmostEqual(positions[0], 0.0)


class FakePort:
    def clearPort(self):
        pass


class FakeBus:
    def __init__(self, positions):
        self.positions = positions
        self.goal_write = None
        self._port = FakePort()

    def read_present_position_tick(self, motor_id):
        return self.positions[motor_id]

    def sync_write_goal_ticks(self, ids, ticks):
        self.goal_write = (list(ids), list(ticks))


if __name__ == "__main__":
    unittest.main()
