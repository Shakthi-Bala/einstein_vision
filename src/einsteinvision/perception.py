"""Perception pipeline orchestration for frame-level scene understanding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class BoundingBox2D:
    """Axis-aligned pixel bounding box in image coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(slots=True)
class DetectedObject2D:
    """2D object detection output for a single tracked entity."""

    track_id: str
    class_name: str
    confidence: float
    bbox: BoundingBox2D
    orientation_yaw_rad: float | None = None


@dataclass(slots=True)
class LaneSegmentationOutput:
    """Lane segmentation result for one frame."""

    lane_mask: np.ndarray
    confidence_map: np.ndarray | None = None


@dataclass(slots=True)
class DepthEstimationOutput:
    """Monocular depth estimation output for one frame."""

    depth_map_m: np.ndarray
    scale_hint: float | None = None


@dataclass(slots=True)
class OpticalFlowOutput:
    """Optical flow output between consecutive frames."""

    flow_uv: np.ndarray
    moving_mask: np.ndarray | None = None


@dataclass(slots=True)
class PerceptionFrameOutput:
    """Unified output of all perception stages for a single frame."""

    frame_index: int
    timestamp_s: float
    lanes: LaneSegmentationOutput | None
    objects: list[DetectedObject2D] = field(default_factory=list)
    depth: DepthEstimationOutput | None = None
    optical_flow: OpticalFlowOutput | None = None


class LaneSegmenter(Protocol):
    """Protocol for pluggable lane segmentation backends."""

    def predict(self, frame_bgr: np.ndarray) -> LaneSegmentationOutput:
        """Runs lane segmentation on a frame."""


class ObjectDetector(Protocol):
    """Protocol for pluggable object detector backends."""

    def predict(self, frame_bgr: np.ndarray) -> list[DetectedObject2D]:
        """Runs object detection/tracking on a frame."""


class DepthEstimator(Protocol):
    """Protocol for pluggable monocular depth estimators."""

    def predict(self, frame_bgr: np.ndarray) -> DepthEstimationOutput:
        """Runs depth estimation on a frame."""


class OpticalFlowEstimator(Protocol):
    """Protocol for pluggable optical flow estimators."""

    def predict(self, prev_frame_bgr: np.ndarray, frame_bgr: np.ndarray) -> OpticalFlowOutput:
        """Computes optical flow from previous frame to current frame."""


class PerceptionPipeline:
    """Orchestrates perception model execution and aggregates outputs."""

    def __init__(
        self,
        lane_segmenter: LaneSegmenter | None = None,
        object_detector: ObjectDetector | None = None,
        depth_estimator: DepthEstimator | None = None,
        optical_flow_estimator: OpticalFlowEstimator | None = None,
    ) -> None:
        self.lane_segmenter = lane_segmenter
        self.object_detector = object_detector
        self.depth_estimator = depth_estimator
        self.optical_flow_estimator = optical_flow_estimator
        self._previous_frame: np.ndarray | None = None

    def run_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_s: float,
    ) -> PerceptionFrameOutput:
        """Runs all configured perception modules on a single frame."""

        lanes = self.segment_lanes(frame_bgr) if self.lane_segmenter else None
        objects = self.detect_objects(frame_bgr) if self.object_detector else []
        depth = self.estimate_depth(frame_bgr) if self.depth_estimator else None

        optical_flow: OpticalFlowOutput | None = None
        if self.optical_flow_estimator and self._previous_frame is not None:
            optical_flow = self.estimate_optical_flow(self._previous_frame, frame_bgr)

        self._previous_frame = frame_bgr
        return PerceptionFrameOutput(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            lanes=lanes,
            objects=objects,
            depth=depth,
            optical_flow=optical_flow,
        )

    def segment_lanes(self, frame_bgr: np.ndarray) -> LaneSegmentationOutput:
        """Delegates lane segmentation to the configured lane model."""

        if self.lane_segmenter is None:
            raise RuntimeError("No lane segmenter configured.")
        return self.lane_segmenter.predict(frame_bgr)

    def detect_objects(self, frame_bgr: np.ndarray) -> list[DetectedObject2D]:
        """Delegates object detection to the configured detector model."""

        if self.object_detector is None:
            raise RuntimeError("No object detector configured.")
        return self.object_detector.predict(frame_bgr)

    def estimate_depth(self, frame_bgr: np.ndarray) -> DepthEstimationOutput:
        """Delegates depth estimation to the configured depth model."""

        if self.depth_estimator is None:
            raise RuntimeError("No depth estimator configured.")
        return self.depth_estimator.predict(frame_bgr)

    def estimate_optical_flow(
        self,
        prev_frame_bgr: np.ndarray,
        frame_bgr: np.ndarray,
    ) -> OpticalFlowOutput:
        """Delegates optical flow computation to the configured flow model."""

        if self.optical_flow_estimator is None:
            raise RuntimeError("No optical flow estimator configured.")
        return self.optical_flow_estimator.predict(prev_frame_bgr, frame_bgr)
