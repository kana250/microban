import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MOTOR_IDS = [11, 21]
_MAIN_MODULE = None


def load_main_module():
    global _MAIN_MODULE
    if _MAIN_MODULE is not None:
        return _MAIN_MODULE

    constants = types.ModuleType("constants")
    constants.MOTOR_TO_ID = {"left_hip_yaw": 11, "right_hip_yaw": 21}
    constants.NEUTRAL_POSE = {"left_hip_yaw": 0.0, "right_hip_yaw": 0.0}
    constants.KP_DEFAULT = 400

    robot_controller = types.ModuleType("robot_controller")
    robot_controller.RobotController = object

    scheduler = types.ModuleType("scheduler")
    scheduler.Scheduler = object

    input_source = types.ModuleType("input.input_source")
    input_source.InputSource = object
    keyboard_input = types.ModuleType("input.keyboard_input")
    keyboard_input.KeyboardInputSource = object

    stub_modules = {
        "constants": constants,
        "robot_controller": robot_controller,
        "scheduler": scheduler,
        "input": types.ModuleType("input"),
        "input.input_source": input_source,
        "input.keyboard_input": keyboard_input,
    }
    for name in ("rotate_head", "squat", "walk"):
        package_name = f"moves.{name}"
        module = types.ModuleType(package_name)
        class_name = {
            "rotate_head": "RotateHeadMove",
            "squat": "SquatMove",
            "walk": "WalkMove",
        }[name]

        class StubMove:
            def __init__(self, *args, **kwargs):
                pass

        setattr(module, class_name, StubMove)
        stub_modules[package_name] = module
    stub_modules["moves"] = types.ModuleType("moves")

    with patch.dict(sys.modules, stub_modules):
        spec = importlib.util.spec_from_file_location("microban_test_main", SRC / "main.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    _MAIN_MODULE = module
    return _MAIN_MODULE


class FakeController:
    def __init__(self, events, fail_hold=False):
        self.events = events
        self.fail_hold = fail_hold

    def sync_write_torque_enable(self, ids, values):
        self.events.append(("torque", tuple(values)))

    def hold_present_position(self, ids):
        self.events.append(("hold", tuple(ids)))
        if self.fail_hold:
            raise RuntimeError("position read failed")
        return [0.1, -0.1]

    def sync_write_status_return_level(self, ids, values):
        self.events.append(("status", tuple(values)))

    def sync_write_kp(self, ids, values):
        self.events.append(("kp", tuple(values)))

    def sync_read_present_position(self, ids):
        return [0.1, -0.1]

    def shutdown(self):
        self.events.append(("shutdown",))


class StartupSafetyTests(unittest.TestCase):
    def test_goal_is_primed_before_torque_enable(self):
        module = load_main_module()
        events = []
        controller = FakeController(events)

        class FakeScheduler:
            def __init__(self, **kwargs):
                self.registered_moves = {}

            def run(self):
                events.append(("run",))

        with tempfile.TemporaryDirectory() as temp_dir:
            module.PID_FILE = Path(temp_dir) / "microban.pid"
            module.RobotController = lambda: controller
            module.Scheduler = FakeScheduler
            module.ramp_to_neutral = lambda ctl, initial_positions=None: events.append(
                ("ramp", tuple(initial_positions))
            )
            module.build_input_source = lambda: object()
            module.main()

        self.assertEqual(events[0], ("torque", (False, False)))
        self.assertEqual(events[1], ("status", (1, 1)))
        self.assertEqual(events[2], ("hold", (11, 21)))
        enable_index = events.index(("torque", (True, True)))
        self.assertLess(events.index(("hold", (11, 21))), enable_index)
        self.assertLess(enable_index, events.index(("ramp", (0.1, -0.1))))
        self.assertEqual(events[-2], ("torque", (False, False)))
        self.assertEqual(events[-1], ("shutdown",))

    def test_position_read_failure_aborts_before_torque_enable(self):
        module = load_main_module()
        events = []
        controller = FakeController(events, fail_hold=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            module.PID_FILE = Path(temp_dir) / "microban.pid"
            module.RobotController = lambda: controller
            with self.assertRaisesRegex(RuntimeError, "position read failed"):
                module.main()

        self.assertNotIn(("torque", (True, True)), events)
        self.assertEqual(events[-2], ("torque", (False, False)))
        self.assertEqual(events[-1], ("shutdown",))


if __name__ == "__main__":
    unittest.main()
