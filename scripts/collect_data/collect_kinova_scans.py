#!/usr/bin/env python3
"""Supervise repeated Isaac Sim, MoveIt, motion-server, and scan runs."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Sequence


ISAAC_READY = (
    "[segmentation_service] Ready: /save_object_segmentations; camera frames: "
    "['handeye_camera_color_optical_frame']"
)


@dataclass(frozen=True)
class CollectionPaths:
    """Resolved external workspaces and repository executables."""

    repository: Path
    kinova_workspace: Path
    isaac_project: Path
    isaac_ros_setup: Path

    @property
    def kinova_setup(self) -> Path:
        return self.kinova_workspace / "install/local_setup.bash"

    @property
    def isaac_launcher(self) -> Path:
        return self.isaac_project / "run_kinova_isaac.sh"

    @property
    def scanner(self) -> Path:
        return self.repository / "scripts/path_planning_single/single_view_scan.py"


class ManagedProcess:
    """Run one terminal command in its own process group and continuously log it."""

    def __init__(
        self,
        name: str,
        command: str,
        log_path: Path,
        *,
        interrupt_process_group: bool = False,
    ) -> None:
        self.name = name
        self.command = command
        self.log_path = log_path
        self.interrupt_process_group = interrupt_process_group
        self.process: subprocess.Popen[str] | None = None
        self._log = None
        self._reader: threading.Thread | None = None
        self._lines: deque[str] = deque(maxlen=500)
        self._condition = threading.Condition()

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8", buffering=1)
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        self.process = subprocess.Popen(
            ["/bin/bash", "--noprofile", "--norc", "-c", self.command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            env=environment,
            start_new_session=True,
        )
        self._reader = threading.Thread(target=self._copy_output, daemon=True)
        self._reader.start()

    def _copy_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        assert self._log is not None
        for line in self.process.stdout:
            self._log.write(line)
            print(f"[{self.name}] {line}", end="", flush=True)
            with self._condition:
                self._lines.append(line.rstrip())
                self._condition.notify_all()
        with self._condition:
            self._condition.notify_all()

    def wait_for_output(self, text: str, timeout: float) -> None:
        """Wait until a readiness marker appears or the process exits."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if any(text in line for line in self._lines):
                    return
                if self.poll() is not None:
                    raise RuntimeError(
                        f"{self.name} exited before readiness (code {self.poll()}); "
                        f"see {self.log_path}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"timed out waiting for {self.name}; see {self.log_path}"
                    )
                self._condition.wait(timeout=min(0.5, remaining))

    def poll(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def wait(self) -> int:
        if self.process is None:
            raise RuntimeError(f"{self.name} was not started")
        return self.process.wait()

    def stop(self, timeout: float) -> None:
        """Request graceful shutdown, escalating across the process group if needed."""
        if self.process is None:
            return
        if self.process.poll() is None:
            # ROS launch must receive SIGINT first so it can stop its children in
            # dependency order. Isaac's shell wrapper instead needs a group signal.
            if self.interrupt_process_group:
                os.killpg(self.process.pid, signal.SIGINT)
            else:
                self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
        if self._reader is not None:
            self._reader.join(timeout=2.0)
        if self._log is not None:
            self._log.close()


def shell_source(path: Path) -> str:
    return f"source {shlex.quote(str(path))}"


def wait_for_ros_service(
    process: ManagedProcess,
    service_name: str,
    expected_type: str,
    kinova_setup: Path,
    timeout: float,
) -> None:
    """Poll ROS discovery while also detecting failure of the owning launch."""
    deadline = time.monotonic() + timeout
    command = (
        f"{shell_source(Path('/opt/ros/humble/setup.bash'))} && "
        f"{shell_source(kinova_setup)} && "
        f"ROS2CLI_NO_DAEMON=1 ros2 service type {shlex.quote(service_name)}"
    )
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{process.name} exited before {service_name} became ready; "
                f"see {process.log_path}"
            )
        try:
            result = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc", "-c", command],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except subprocess.TimeoutExpired:
            result = None
        if result is not None and expected_type in result.stdout:
            return
        time.sleep(1.0)
    raise TimeoutError(
        f"timed out waiting for ROS service {service_name} ({expected_type}); "
        f"see {process.log_path}"
    )


def wait_for_joint_state(
    process: ManagedProcess,
    kinova_setup: Path,
    timeout: float,
) -> None:
    """Require one live joint state before starting the motion interface."""
    deadline = time.monotonic() + timeout
    command = (
        f"{shell_source(Path('/opt/ros/humble/setup.bash'))} && "
        f"{shell_source(kinova_setup)} && ROS2CLI_NO_DAEMON=1 "
        "timeout 4s ros2 topic echo /joint_states sensor_msgs/msg/JointState "
        "--once --field name"
    )
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{process.name} exited before /joint_states became ready; "
                f"see {process.log_path}"
            )
        try:
            result = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc", "-c", command],
                capture_output=True,
                text=True,
                timeout=6.0,
                check=False,
            )
        except subprocess.TimeoutExpired:
            result = None
        if result is not None and result.returncode == 0 and result.stdout.strip():
            return
        time.sleep(1.0)
    raise TimeoutError(
        f"timed out waiting for a live /joint_states message; see {process.log_path}"
    )


def wait_for_active_controllers(
    process: ManagedProcess,
    kinova_setup: Path,
    controller_names: Sequence[str],
    timeout: float,
) -> None:
    """Wait until ros2_control reports every required controller as active."""
    deadline = time.monotonic() + timeout
    command = (
        f"{shell_source(Path('/opt/ros/humble/setup.bash'))} && "
        f"{shell_source(kinova_setup)} && ROS2CLI_NO_DAEMON=1 "
        "ros2 control list_controllers --controller-manager /controller_manager"
    )
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{process.name} exited before ros2_control became ready; "
                f"see {process.log_path}"
            )
        try:
            result = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc", "-c", command],
                capture_output=True,
                text=True,
                timeout=6.0,
                check=False,
            )
        except subprocess.TimeoutExpired:
            result = None
        if result is not None and result.returncode == 0:
            active = {
                line.split()[0]
                for line in result.stdout.splitlines()
                if line.split() and line.split()[-1] == "active"
            }
            if set(controller_names).issubset(active):
                return
        time.sleep(1.0)
    names = ", ".join(controller_names)
    raise TimeoutError(
        f"timed out waiting for active controller(s) {names}; see {process.log_path}"
    )


def validate_paths(paths: CollectionPaths) -> None:
    """Fail before launching anything when a required workspace file is absent."""
    required = (
        Path("/opt/ros/humble/setup.bash"),
        paths.kinova_setup,
        paths.isaac_ros_setup,
        paths.isaac_launcher,
        paths.scanner,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required files:\n  " + "\n  ".join(missing))


def commands(paths: CollectionPaths, robot_ip: str, suffix: str) -> dict[str, str]:
    """Build the four commands in the exact required environment order."""
    ros = shell_source(Path("/opt/ros/humble/setup.bash"))
    kinova = shell_source(paths.kinova_setup)
    clean_ros_environment = (
        "unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH "
        "PYTHONPATH ROS_PACKAGE_PATH PKG_CONFIG_PATH LD_LIBRARY_PATH "
        "RMW_IMPLEMENTATION ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION"
    )
    return {
        "isaac": (
            f"set -e; {clean_ros_environment}; "
            f"{shell_source(paths.isaac_ros_setup)}; "
            'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#:}"; '
            f"cd {shlex.quote(str(paths.isaac_project))}; "
            f"exec /bin/bash {shlex.quote(str(paths.isaac_launcher))}"
        ),
        "moveit": (
            f"set -e; {ros}; {kinova}; exec ros2 launch "
            "kinova_gen3_7dof_robotiq_2f_85_moveit_config robot.launch.py "
            f"robot_ip:={shlex.quote(robot_ip)} isaac_sim:=true launch_rviz:=false"
        ),
        "motion": (
            f"set -e; {ros}; {kinova}; exec ros2 launch robot_interfaces "
            "robot_interfaces.launch.py start_gripper_server:=false "
            "default_planning_group:=manipulator "
            "tracking_tip_links:=end_effector_link velocity_scale:=0.5 "
            "acceleration_scale:=0.5"
        ),
        "scan": (
            f"set -e; {ros}; {kinova}; cd {shlex.quote(str(paths.repository))}; "
            f"exec python3 {shlex.quote(str(paths.scanner))} --use-sim-time "
            f"--output-suffix {shlex.quote(suffix)}"
        ),
    }


def run_once(args: argparse.Namespace, paths: CollectionPaths, run_number: int) -> dict:
    """Launch one complete four-stage collection and always tear it down."""
    suffix = f"{args.output_prefix}_{run_number}"
    log_directory = paths.repository / args.log_directory / suffix
    run_commands = commands(paths, args.robot_ip, suffix)
    active: list[ManagedProcess] = []
    started_at = datetime.now(timezone.utc)
    status = "failed"
    error = None

    try:
        # Terminal 1: wait for the exact segmentation readiness contract.
        isaac = ManagedProcess(
            "isaac",
            run_commands["isaac"],
            log_directory / "isaac.log",
            interrupt_process_group=True,
        )
        active.append(isaac)
        isaac.start()
        isaac.wait_for_output(ISAAC_READY, args.isaac_timeout)
        time.sleep(3.0)  # allow Isaac to finish its own startup before MoveIt

        # Terminal 2: /compute_ik proves MoveIt's kinematics service is discoverable.
        moveit = ManagedProcess(
            "moveit", run_commands["moveit"], log_directory / "moveit.log"
        )
        active.append(moveit)
        moveit.start()
        wait_for_ros_service(
            moveit,
            "/compute_ik",
            "moveit_msgs/srv/GetPositionIK",
            paths.kinova_setup,
            args.moveit_timeout,
        )
        wait_for_active_controllers(
            moveit,
            paths.kinova_setup,
            ("joint_state_broadcaster", "joint_trajectory_controller"),
            args.moveit_timeout,
        )
        wait_for_joint_state(moveit, paths.kinova_setup, args.moveit_timeout)
        time.sleep(3.0)  # allow MoveIt to finish its own startup before motion-server

        # Terminal 3: wait for the RobotAPI Cartesian service before scanning.
        motion = ManagedProcess(
            "motion", run_commands["motion"], log_directory / "motion.log"
        )
        active.append(motion)
        motion.start()
        wait_for_ros_service(
            motion,
            "/move_cartesian",
            "robot_interfaces/srv/Move3DPose",
            paths.kinova_setup,
            args.motion_timeout,
        )
        time.sleep(3.0)  # allow motion-server to finish its own startup before scanning

        # Terminal 4 is finite; its exit code determines collection success.
        scan = ManagedProcess("scan", run_commands["scan"], log_directory / "scan.log")
        active.append(scan)
        scan.start()
        scan_code = scan.wait()
        if scan_code != 0:
            raise RuntimeError(f"scan exited with code {scan_code}; see {scan.log_path}")
        status = "completed"
    except (OSError, RuntimeError, TimeoutError) as exception:
        error = str(exception)
        print(f"[collector] Run {run_number} failed: {error}", file=sys.stderr)
    finally:
        # Stop in reverse dependency order before creating the next random scene.
        for process in reversed(active):
            try:
                process.stop(args.shutdown_timeout)
            except (OSError, subprocess.SubprocessError) as exception:
                print(
                    f"[collector] Failed to stop {process.name}: {exception}",
                    file=sys.stderr,
                )

    record = {
        "run_number": run_number,
        "output_suffix": suffix,
        "status": status,
        "error": error,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "log_directory": str(log_directory),
    }
    log_directory.mkdir(parents=True, exist_ok=True)
    (log_directory / "run.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, required=True, help="number of runs")
    parser.add_argument("--start-run", type=int, required=True, help="first run number")
    parser.add_argument("--robot-ip", default="0.0.0.0")
    parser.add_argument("--output-prefix", default="run")
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--kinova-workspace", type=Path, default=Path("~/Projects/kinova_ws"))
    parser.add_argument("--isaac-project", type=Path, default=Path("~/Projects/kinova_isaacsim"))
    parser.add_argument("--isaac-ros-setup", type=Path, default=Path("~/isaac_ros.sh"))
    parser.add_argument("--log-directory", type=Path, default=Path("scan_output/collection_logs"))
    parser.add_argument("--isaac-timeout", type=float, default=300.0)
    parser.add_argument("--moveit-timeout", type=float, default=180.0)
    parser.add_argument("--motion-timeout", type=float, default=120.0)
    parser.add_argument("--shutdown-timeout", type=float, default=30.0)
    parser.add_argument("--restart-delay", type=float, default=5.0)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands without starting processes"
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.start_run < 0:
        raise ValueError("--start-run must be nonnegative")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.output_prefix):
        raise ValueError("--output-prefix must be a safe non-empty filename prefix")
    timeouts = (
        args.isaac_timeout,
        args.moveit_timeout,
        args.motion_timeout,
        args.shutdown_timeout,
    )
    if any(value <= 0.0 for value in timeouts) or args.restart_delay < 0.0:
        raise ValueError("timeouts must be positive and restart delay nonnegative")


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    validate_arguments(args)
    paths = CollectionPaths(
        repository=args.repository.expanduser().resolve(),
        kinova_workspace=args.kinova_workspace.expanduser().resolve(),
        isaac_project=args.isaac_project.expanduser().resolve(),
        isaac_ros_setup=args.isaac_ros_setup.expanduser().resolve(),
    )
    validate_paths(paths)

    if args.dry_run:
        for run_number in range(args.start_run, args.start_run + args.runs):
            suffix = f"{args.output_prefix}_{run_number}"
            print(f"Run {run_number} ({suffix}):")
            for terminal, command in commands(paths, args.robot_ip, suffix).items():
                print(f"  {terminal}: {command}")
        return 0

    records = []
    try:
        for offset in range(args.runs):
            run_number = args.start_run + offset
            print(f"[collector] Starting run {run_number} ({offset + 1}/{args.runs})")
            record = run_once(args, paths, run_number)
            records.append(record)
            if record["status"] != "completed" and args.stop_on_error:
                break
            if offset + 1 < args.runs:
                time.sleep(args.restart_delay)
    except KeyboardInterrupt:
        print("\n[collector] Interrupted; active processes are shutting down.")
        return 130

    completed = sum(record["status"] == "completed" for record in records)
    print(f"[collector] Completed {completed}/{len(records)} attempted runs")
    return 0 if completed == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
