"""Client for Isaac Sim's on-demand object-segmentation capture service."""

from __future__ import annotations

import math
from pathlib import Path

import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParametersAtomically
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter as NodeParameter


class SegmentationServiceError(RuntimeError):
    """Raised when Isaac Sim cannot save a requested segmentation capture."""


class IsaacSegmentationClient:
    """Own a private ROS context for synchronous segmentation service calls."""

    def __init__(
        self,
        service_name: str,
        *,
        use_sim_time: bool = True,
    ) -> None:
        if not service_name:
            raise ValueError("segmentation service name must not be empty")
        self._context = Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node(
            "multi_view_scan_segmentation",
            context=self._context,
            parameter_overrides=[NodeParameter("use_sim_time", value=use_sim_time)],
            automatically_declare_parameters_from_overrides=True,
        )
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._client = self._node.create_client(
            SetParametersAtomically, service_name
        )
        self._service_name = service_name
        self._closed = False

    def __enter__(self) -> "IsaacSegmentationClient":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def wait_for_service(self, timeout: float) -> None:
        """Fail before robot motion when Isaac's segmentation service is absent."""
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("segmentation timeout must be finite and positive")
        if not self._client.wait_for_service(timeout_sec=timeout):
            raise SegmentationServiceError(
                f"segmentation service unavailable: {self._service_name}"
            )

    def save(self, save_directory: Path, camera_frame: str, timeout: float) -> Path:
        """Request one camera capture and return Isaac's generated directory."""
        if not camera_frame:
            raise ValueError("segmentation camera frame must not be empty")
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("segmentation timeout must be finite and positive")

        request = SetParametersAtomically.Request()
        request.parameters = [
            self._string_parameter("save_directory", str(save_directory.resolve())),
            self._string_parameter("camera_frame", camera_frame),
        ]
        future = self._client.call_async(request)
        self._executor.spin_until_future_complete(future, timeout_sec=timeout)
        if not future.done():
            raise SegmentationServiceError(
                f"segmentation request timed out after {timeout:.1f} seconds"
            )
        if future.exception() is not None:
            raise SegmentationServiceError(str(future.exception()))
        response = future.result()
        if response is None or not response.result.successful:
            reason = response.result.reason if response is not None else "no response"
            raise SegmentationServiceError(reason)

        prefix = "saved segmentation to "
        reason = response.result.reason
        if not reason.startswith(prefix):
            raise SegmentationServiceError(
                f"segmentation service returned no output directory: {reason}"
            )
        capture_directory = Path(reason[len(prefix):]).resolve()
        return capture_directory

    @staticmethod
    def _string_parameter(name: str, value: str) -> Parameter:
        return Parameter(
            name=name,
            value=ParameterValue(
                type=ParameterType.PARAMETER_STRING,
                string_value=value,
            ),
        )

    def close(self) -> None:
        """Release the private executor, node, and ROS context."""
        if self._closed:
            return
        self._closed = True
        self._executor.remove_node(self._node)
        self._node.destroy_node()
        self._executor.shutdown()
        if self._context.ok():
            self._context.shutdown()
