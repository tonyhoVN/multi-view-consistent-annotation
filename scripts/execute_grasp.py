#!/usr/bin/env python3
"""Validate and execute a sampled grasp command with ``RobotAPI``.

The command files are produced by :mod:`grasp_sampling` and normally live in
``scan_output/grasp_commands``.  This script is a dry run unless ``--execute``
is supplied, so inspecting a newly generated grasp cannot move the robot.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_ACTIONS = (
    "open_gripper",
    "move_cartesian",
    "move_cartesian",
    "close_gripper",
)
ALLOWED_PLANNING_GROUPS = {
    # "left_arm",
    # "right_arm",
    "dual_arm",
    # "whole_body",
}


class CommandValidationError(ValueError):
    """Raised when a command file is malformed or internally inconsistent."""


class GraspExecutionError(RuntimeError):
    """Raised when RobotAPI fails before the grasp sequence is complete."""


@dataclass(frozen=True)
class CartesianPose:
    """A validated Cartesian pose in the world frame."""

    position_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class GraspStep:
    """One validated gripper or Cartesian command."""

    action: str
    end_effector: str | None = None
    planning_group: str | None = None
    pose: CartesianPose | None = None


@dataclass(frozen=True)
class GraspPlan:
    """A complete grasp plan ready for dry-run display or execution."""

    object_name: str
    arm: str
    end_effector: str
    planning_group: str
    score: float
    required_opening_m: float
    steps: tuple[GraspStep, ...]


def _finite_sequence(
    value: Any, *, length: int, field_name: str
) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CommandValidationError(f"{field_name} must be a sequence")
    if len(value) != length:
        raise CommandValidationError(
            f"{field_name} must contain exactly {length} values"
        )
    try:
        numbers = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise CommandValidationError(f"{field_name} must contain numbers") from error
    if not all(math.isfinite(number) for number in numbers):
        raise CommandValidationError(f"{field_name} must contain only finite values")
    return numbers


def _validate_pose(value: Any, *, step_number: int) -> CartesianPose:
    if not isinstance(value, Mapping):
        raise CommandValidationError(f"step {step_number} pose must be an object")
    position = _finite_sequence(
        value.get("position_xyz"),
        length=3,
        field_name=f"step {step_number} position_xyz",
    )
    quaternion = _finite_sequence(
        value.get("quaternion_xyzw"),
        length=4,
        field_name=f"step {step_number} quaternion_xyzw",
    )
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm < 1e-8:
        raise CommandValidationError(f"step {step_number} quaternion is zero")
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise CommandValidationError(
            f"step {step_number} quaternion must be normalized (norm={norm:.6f})"
        )
    return CartesianPose(
        position_xyz=(position[0], position[1], position[2]),
        quaternion_xyzw=(quaternion[0], quaternion[1], quaternion[2], quaternion[3]),
    )


def validate_grasp_plan(
    value: Any,
    *,
    minimum_score: float = 0.0,
    maximum_opening_m: float = 0.08,
) -> GraspPlan:
    """Validate a ``best_command`` record and return a typed execution plan."""
    if not isinstance(value, Mapping):
        raise CommandValidationError("grasp command must be a JSON object")

    object_name = value.get("object", "object")
    if not isinstance(object_name, str) or not object_name.strip():
        raise CommandValidationError("object must be a non-empty string")

    arm = value.get("arm")
    if arm not in {"left", "right"}:
        raise CommandValidationError("arm must be 'left' or 'right'")
    expected_end_effector = f"{arm}_ee"
    end_effector = value.get("end_effector")
    if end_effector != expected_end_effector:
        raise CommandValidationError(
            f"end_effector must be {expected_end_effector!r} for the {arm} arm"
        )

    planning_group = value.get("planning_group")
    if planning_group not in ALLOWED_PLANNING_GROUPS:
        allowed = ", ".join(sorted(ALLOWED_PLANNING_GROUPS))
        raise CommandValidationError(f"planning_group must be one of: {allowed}")
    opposite_group = "right_arm" if arm == "left" else "left_arm"
    if planning_group == opposite_group:
        raise CommandValidationError(
            f"planning_group {planning_group!r} does not control the {arm} arm"
        )

    try:
        score = float(value.get("score"))
        required_opening = float(value.get("required_opening_m"))
    except (TypeError, ValueError) as error:
        raise CommandValidationError(
            "score and required_opening_m must be numbers"
        ) from error
    if not math.isfinite(score) or score < minimum_score:
        raise CommandValidationError(
            f"grasp score {score!r} is below the required {minimum_score:.3f}"
        )
    if not math.isfinite(required_opening) or not (
        0.0 < required_opening <= maximum_opening_m
    ):
        raise CommandValidationError(
            "required_opening_m must be greater than zero and no more than "
            f"{maximum_opening_m:.3f} m"
        )

    commands = value.get("commands")
    if not isinstance(commands, list):
        raise CommandValidationError("commands must be a list")
    actions = tuple(
        command.get("action") if isinstance(command, Mapping) else None
        for command in commands
    )
    if actions != EXPECTED_ACTIONS:
        raise CommandValidationError(
            "commands must have this exact safe sequence: "
            + " -> ".join(EXPECTED_ACTIONS)
        )

    steps: list[GraspStep] = []
    for step_number, command in enumerate(commands, start=1):
        assert isinstance(command, Mapping)  # established by the action check
        action = str(command["action"])
        if action in {"open_gripper", "close_gripper"}:
            command_ee = command.get("end_effector")
            if command_ee != end_effector:
                raise CommandValidationError(
                    f"step {step_number} must address {end_effector!r}"
                )
            steps.append(GraspStep(action=action, end_effector=end_effector))
            continue

        command_group = command.get("planning_group")
        if command_group != planning_group:
            raise CommandValidationError(
                f"step {step_number} planning_group differs from the plan"
            )
        steps.append(
            GraspStep(
                action=action,
                planning_group=planning_group,
                pose=_validate_pose(command.get("pose"), step_number=step_number),
            )
        )

    grasp_pose = steps[2].pose
    assert grasp_pose is not None
    arm_from_world_y = "left" if grasp_pose.position_xyz[1] >= 0.0 else "right"
    if arm != arm_from_world_y:
        raise CommandValidationError(
            f"grasp world y={grasp_pose.position_xyz[1]:.6f} requires the "
            f"{arm_from_world_y} arm, but the command selects the {arm} arm"
        )

    return GraspPlan(
        object_name=object_name.strip(),
        arm=arm,
        end_effector=end_effector,
        planning_group=planning_group,
        score=score,
        required_opening_m=required_opening,
        steps=tuple(steps),
    )


def load_grasp_plan(
    path: Path,
    *,
    minimum_score: float = 0.0,
    maximum_opening_m: float = 0.08,
) -> GraspPlan:
    """Load either a command record or a sampler output containing one."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CommandValidationError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise CommandValidationError(f"invalid JSON in {path}: {error}") from error
    if isinstance(document, Mapping) and "best_command" in document:
        document = document["best_command"]
    return validate_grasp_plan(
        document,
        minimum_score=minimum_score,
        maximum_opening_m=maximum_opening_m,
    )


def print_plan(plan: GraspPlan) -> None:
    """Print the exact operations that would be sent to RobotAPI."""
    print(
        f"Object: {plan.object_name}\n"
        f"Arm: {plan.arm} ({plan.end_effector}, group={plan.planning_group})\n"
        f"Score: {plan.score:.3f}\n"
        f"Required opening: {plan.required_opening_m * 1000.0:.1f} mm"
    )
    for number, step in enumerate(plan.steps, start=1):
        if step.pose is None:
            print(f"  {number}. {step.action}({step.end_effector})")
        else:
            xyz = ", ".join(f"{value:.4f}" for value in step.pose.position_xyz)
            xyzw = ", ".join(f"{value:.5f}" for value in step.pose.quaternion_xyzw)
            print(
                f"  {number}. move_cartesian({plan.end_effector}, "
                f"xyz=[{xyz}], xyzw=[{xyzw}])"
            )


def execute_grasp_plan(
    plan: GraspPlan,
    *,
    call_timeout: float = 15.0,
    robot_api_class: Any = None,
    use_sim_time: bool = False,
) -> None:
    """Execute a validated plan, aborting immediately if any step raises."""
    if call_timeout <= 0.0:
        raise ValueError("call_timeout must be greater than zero")

    if robot_api_class is None:
        # Import lazily so validation and dry runs also work outside a ROS setup.
        from robot_api import RobotAPI

        robot_api_class = RobotAPI

    with robot_api_class(
        node_name="grasp_command_executor",
        call_timeout=call_timeout,
        use_sim_time=use_sim_time,
    ) as robot:
        for step_number, step in enumerate(plan.steps, start=1):
            print(f"Executing {step_number}/{len(plan.steps)}: {step.action}")
            try:
                if step.action == "open_gripper":
                    result = robot.open_gripper(
                        step.end_effector, timeout=call_timeout
                    )
                elif step.action == "close_gripper":
                    result = robot.close_gripper(
                        step.end_effector, timeout=call_timeout
                    )
                else:
                    assert step.pose is not None
                    target = robot_api_class.pose(
                        step.pose.position_xyz, step.pose.quaternion_xyzw
                    )
                    result = robot.move_cartesian(
                        {plan.end_effector: target},
                        planning_group=step.planning_group,
                        relative=False,
                        timeout=call_timeout,
                    )
            except Exception as error:
                raise GraspExecutionError(
                    f"execution aborted at step {step_number} ({step.action}): {error}"
                ) from error
            message = getattr(result, "message", "")
            print(f"  completed{': ' + message if message else ''}")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or execute a sampled grasp command with RobotAPI."
    )
    parser.add_argument(
        "command_file",
        type=Path,
        help="JSON file containing best_command or a direct grasp command",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="send commands to the robot (the default is a dry run)",
    )
    parser.add_argument(
        "--minimum-score",
        type=float,
        default=0.0,
        help="reject grasps below this score (default: 0.0)",
    )
    parser.add_argument(
        "--maximum-opening",
        type=float,
        default=0.08,
        metavar="METERS",
        help="reject grasps wider than this limit (default: 0.08)",
    )
    parser.add_argument(
        "--call-timeout",
        type=float,
        default=120.0,
        metavar="SECONDS",
        help="RobotAPI timeout for each command (default: 120)",
    )
    return parser


def main() -> int:
    """CLI entry point."""
    args = _argument_parser().parse_args()
    if not math.isfinite(args.minimum_score):
        raise SystemExit("--minimum-score must be finite")
    if not math.isfinite(args.maximum_opening) or args.maximum_opening <= 0.0:
        raise SystemExit("--maximum-opening must be finite and greater than zero")
    if not math.isfinite(args.call_timeout) or args.call_timeout <= 0.0:
        raise SystemExit("--call-timeout must be finite and greater than zero")

    try:
        plan = load_grasp_plan(
            args.command_file,
            minimum_score=args.minimum_score,
            maximum_opening_m=args.maximum_opening,
        )
        print_plan(plan)
        if not args.execute:
            print("Dry run only. Add --execute to move the robot.")
            return 0
        print("Execution enabled; sending the validated sequence to RobotAPI.")
        execute_grasp_plan(plan, call_timeout=args.call_timeout)
    except (CommandValidationError, GraspExecutionError, ImportError) as error:
        raise SystemExit(f"error: {error}") from error
    print("Grasp command sequence completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
