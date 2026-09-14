#!/usr/bin/env python3
"""Collect Kinova scans after recording exact Isaac scene object identities.

This variant preserves the lifecycle and environment setup of
``collect_kinova_scans.py``. After Isaac becomes ready, it calls the Trigger
service ``/list_scene_objects`` before any robot motion or image capture.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import collect_kinova_scans as collector  # noqa: E402


def validate_scene_objects(value: Any) -> list[dict[str, str]]:
    """Validate exact class, instance, and prim names returned by Isaac."""
    if not isinstance(value, list) or not value:
        raise ValueError("scene-object service message must be a nonempty JSON array")
    required = ("class_name", "instance_name", "prim_path")
    objects = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"scene object {index} is not a JSON object")
        record = {}
        for field in required:
            field_value = item.get(field)
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"scene object {index} has invalid {field}")
            record[field] = field_value.strip()
        objects.append(record)
    instances = [item["instance_name"] for item in objects]
    if len(instances) != len(set(instances)):
        raise ValueError("scene-object service returned duplicate instance_name values")
    return objects


def parse_trigger_response(output: str) -> list[dict[str, str]]:
    """Extract and parse Trigger.message from ROS 2 CLI response text."""
    if not re.search(r"success\s*=\s*True", output):
        raise RuntimeError(f"/list_scene_objects reported failure: {output.strip()}")
    match = re.search(
        r"message\s*=\s*(?P<message>'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")\s*\)\s*$",
        output,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("cannot extract Trigger.message from ros2 service output")
    message = ast.literal_eval(match.group("message"))
    try:
        document = json.loads(message)
    except json.JSONDecodeError as error:
        raise ValueError(f"/list_scene_objects message is not valid JSON: {error}") from error
    return validate_scene_objects(document)


def request_scene_objects(
    paths: collector.CollectionPaths,
    service_name: str,
    timeout: float,
    raw_log: Path,
) -> list[dict[str, str]]:
    """Call Isaac's Trigger service in a correctly sourced ROS subprocess."""
    command = (
        f"{collector.shell_source(Path('/opt/ros/humble/setup.bash'))} && "
        f"{collector.shell_source(paths.kinova_setup)} && "
        "ROS2CLI_NO_DAEMON=1 "
        f"timeout {timeout:.3f}s ros2 service call "
        f"{shlex.quote(service_name)} std_srvs/srv/Trigger '{{}}'"
    )
    result = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", command],
        capture_output=True,
        text=True,
        timeout=timeout + 5.0,
        check=False,
    )
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    raw_log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"scene-object service call exited with code {result.returncode}; "
            f"see {raw_log}"
        )
    return parse_trigger_response(result.stdout)


def save_scene_objects(
    path: Path, service_name: str, objects: Sequence[dict[str, str]]
) -> None:
    """Persist the pre-scan scene inventory independently of scan success."""
    record = {
        "service": service_name,
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "object_count": len(objects),
        "objects": list(objects),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def attach_scene_objects_to_manifest(
    manifest_path: Path, scene_path: Path, objects: Sequence[dict[str, str]]
) -> None:
    """Add portable scene identity metadata after the scanner saves its manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"scanner produced no manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scene_objects_file"] = scene_path.relative_to(manifest_path.parent).as_posix()
    manifest["scene_objects"] = list(objects)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_once(
    args: argparse.Namespace, paths: collector.CollectionPaths, run_number: int
) -> dict[str, Any]:
    """Run one four-process collection with a pre-motion scene query."""
    suffix = f"{args.output_prefix}_{run_number}"
    scan_output = (paths.repository / args.scan_output_dir).resolve()
    layout = collector.ScanRunLayout(scan_output, suffix)
    log_directory = layout.collection_log
    run_commands = collector.commands(paths, args.robot_ip, suffix, scan_output)
    active: list[collector.ManagedProcess] = []
    started_at = datetime.now(timezone.utc)
    status, error = "failed", None
    objects: list[dict[str, str]] = []
    scene_path = layout.root / "scene_objects.json"

    try:
        # Terminal 1 owns both Isaac segmentation services and stays alive.
        isaac = collector.ManagedProcess(
            "isaac",
            run_commands["isaac"],
            log_directory / "isaac.log",
            interrupt_process_group=True,
        )
        active.append(isaac)
        isaac.start()
        isaac.wait_for_output(collector.ISAAC_READY, args.isaac_timeout)
        collector.wait_for_ros_service(
            isaac,
            args.scene_object_service,
            "std_srvs/srv/Trigger",
            paths.kinova_setup,
            args.scene_object_timeout,
        )
        objects = request_scene_objects(
            paths,
            args.scene_object_service,
            args.scene_object_timeout,
            log_directory / "list_scene_objects.log",
        )
        save_scene_objects(scene_path, args.scene_object_service, objects)
        print(
            "[collector] Scene objects: "
            + ", ".join(item["instance_name"] for item in objects)
        )

        # Start MoveIt only after the immutable scene inventory is recorded.
        moveit = collector.ManagedProcess(
            "moveit", run_commands["moveit"], log_directory / "moveit.log"
        )
        active.append(moveit)
        moveit.start()
        collector.wait_for_ros_service(
            moveit, "/compute_ik", "moveit_msgs/srv/GetPositionIK",
            paths.kinova_setup, args.moveit_timeout,
        )
        collector.wait_for_active_controllers(
            moveit,
            paths.kinova_setup,
            ("joint_state_broadcaster", "joint_trajectory_controller"),
            args.moveit_timeout,
        )
        collector.wait_for_joint_state(moveit, paths.kinova_setup, args.moveit_timeout)

        # Start RobotAPI, then execute the finite scanner in terminal 4.
        motion = collector.ManagedProcess(
            "motion", run_commands["motion"], log_directory / "motion.log"
        )
        active.append(motion)
        motion.start()
        collector.wait_for_ros_service(
            motion, "/move_cartesian", "robot_interfaces/srv/Move3DPose",
            paths.kinova_setup, args.motion_timeout,
        )
        scan = collector.ManagedProcess(
            "scan", run_commands["scan"], log_directory / "scan.log"
        )
        active.append(scan)
        scan.start()
        scan_code = scan.wait()
        if scan_code != 0:
            raise RuntimeError(f"scan exited with code {scan_code}; see {scan.log_path}")
        attach_scene_objects_to_manifest(layout.manifest, scene_path, objects)
        status = "completed"
    except (
        OSError,
        RuntimeError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exception:
        error = str(exception)
        print(f"[collector] Run {run_number} failed: {error}", file=sys.stderr)
    finally:
        for process in reversed(active):
            try:
                process.stop(args.shutdown_timeout)
            except (OSError, subprocess.SubprocessError) as exception:
                print(f"[collector] Failed to stop {process.name}: {exception}", file=sys.stderr)

    record = {
        "run_number": run_number,
        "output_suffix": suffix,
        "status": status,
        "error": error,
        "scene_object_count": len(objects),
        "scene_objects_file": collector.serialized_relative_path(
            scene_path, log_directory
        ),
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "log_directory": collector.serialized_relative_path(log_directory, log_directory),
    }
    log_directory.mkdir(parents=True, exist_ok=True)
    (log_directory / "run.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = collector.build_parser()
    parser.description = __doc__
    parser.add_argument("--scene-object-service", default="/list_scene_objects")
    parser.add_argument("--scene-object-timeout", type=float, default=30.0)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    collector.validate_arguments(args)
    if args.scene_object_timeout <= 0.0:
        raise ValueError("--scene-object-timeout must be positive")
    paths = collector.CollectionPaths(
        repository=args.repository.expanduser().resolve(),
        kinova_workspace=args.kinova_workspace.expanduser().resolve(),
        isaac_project=args.isaac_project.expanduser().resolve(),
        isaac_ros_setup=args.isaac_ros_setup.expanduser().resolve(),
    )
    collector.validate_paths(paths)

    if args.dry_run:
        for run_number in range(args.start_run, args.start_run + args.runs):
            suffix = f"{args.output_prefix}_{run_number}"
            print(f"Run {run_number} ({suffix}):")
            scan_output = (paths.repository / args.scan_output_dir).resolve()
            for terminal, command in collector.commands(
                paths, args.robot_ip, suffix, scan_output
            ).items():
                print(f"  {terminal}: {command}")
            print(
                f"  pre-scan query: {args.scene_object_service} "
                "(std_srvs/srv/Trigger)"
            )
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
