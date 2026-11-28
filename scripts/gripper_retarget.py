"""TacVerse 夹爪单帧重定向。

float32 深度图 → 已标定的固定 MANO 表面（拇指 / 食指各一套像素对应）
→ 官方 UV 上按 texel 平均 → uint8 接触代理、可选 uint16 深度、以及观测 mask。

两指都贴在同一个 778 顶点网格上。现有地图的约定是：

- ``left_tactile_left`` → MANO 拇指指尖区域
- ``left_tactile_right`` → MANO 食指指尖区域

右手复用这份左夹爪地图，是未验证假设，不是第二套标定。接触代理是
``round(255 * clip((depth - 0.001) / (0.4 - 0.001), 0, 1))``，不是压力，
也不是 Sharpa 形变强度。没观测到的 texel 保持未知，不写成 0 接触。

批量解码、tar 发布仍在 ``pipeline/tacverse_retarget`` 与
``scripts/tacverse/convert_tacverse.py``。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np

from sharpa_retarget import BACKGROUND, CONTACT_LUT, UV_DIR, encode_png, official_uv

REPO = Path(__file__).resolve().parent
DEFAULT_MAPPING = REPO / "vendor" / "tacverse" / "fixed" / "maps" / "fixed_mapping.h5"

# 生产默认。深度范围含一个略低于接触阈值的下界，接触代理从 0.001 起算。
DEPTH_MIN = -0.001
DEPTH_MAX = 0.4
CONTACT_MIN = 0.001
CONTACT_MAX = 0.4

FINGERS = ("thumb", "index")
CHANNELS = {
    "left": {"thumb": "left_tactile_left", "index": "left_tactile_right"},
    "right": {"thumb": "right_tactile_left", "index": "right_tactile_right"},
}


class FixedMapping:
    """离线像素 → 面片重心坐标。这里只校验并采样，不重新拟合。"""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or DEFAULT_MAPPING)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with h5py.File(self.path, "r") as handle:
            if handle.attrs["schema"] != "tacverse.fixed_mapping.v1":
                raise ValueError("Unsupported mapping schema")
            self.shape = tuple(handle.attrs["source_shape"])
            self.vertices = handle["mesh/vertices"][:]
            self.faces = handle["mesh/faces"][:]
            self.fingers = {}
            for finger in FINGERS:
                group = handle[finger]
                self.fingers[finger] = {
                    key: group[key][:]
                    for key in (
                        "source_pixel_indices",
                        "face_id",
                        "barycentric",
                        "mapping_valid",
                        "target_xyz",
                    )
                }
                record = self.fingers[finger]
                pixels = record["source_pixel_indices"]
                valid = record["mapping_valid"]
                face_id = record["face_id"]
                barycentric = record["barycentric"]
                if len(pixels) != len(valid) or np.unique(pixels).size != len(pixels):
                    raise ValueError("Invalid source pixel map")
                if np.any(pixels < 0) or np.any(pixels >= np.prod(self.shape)):
                    raise ValueError("Out-of-range source pixel")
                if (
                    not valid.any()
                    or np.any(face_id[valid] < 0)
                    or np.any(face_id[valid] >= len(self.faces))
                ):
                    raise ValueError("Invalid target face")
                if (
                    np.max(np.abs(barycentric[valid].sum(1) - 1)) > 1e-9
                    or barycentric[valid].min() < -1e-8
                ):
                    raise ValueError("Invalid barycentric coordinates")
                reconstructed = np.einsum(
                    "ni,nij->nj",
                    barycentric[valid],
                    self.vertices[self.faces[face_id[valid]]],
                )
                if (
                    np.max(
                        np.linalg.norm(
                            reconstructed - record["target_xyz"][valid], axis=1
                        )
                    )
                    > 1e-8
                ):
                    raise ValueError("Target position does not match fixed mesh")
                record["reference_channel"] = group.attrs["reference_channel"]

    def apply(self, depth: np.ndarray, finger: str) -> np.ndarray:
        image = np.asarray(depth)
        if image.dtype != np.float32 or image.shape != self.shape:
            raise ValueError(
                f"Expected float32 depth shape {self.shape}; received {image.dtype} {image.shape}"
            )
        record = self.fingers[finger]
        values = image.reshape(-1)[record["source_pixel_indices"]]
        valid = record["mapping_valid"] & np.isfinite(values)
        values = values.copy()
        values[~valid] = np.nan
        return values


def transform_depth(depth: np.ndarray, transform: str) -> np.ndarray:
    """右手计划里允许的朝向修正。只翻转，不缩放。"""
    if transform == "flip_x":
        depth = depth[:, ::-1]
    elif transform == "flip_y":
        depth = depth[::-1, :]
    elif transform == "rotate180":
        depth = depth[::-1, ::-1]
    elif transform != "identity":
        raise ValueError(
            "Mapping transform must be identity, flip_x, flip_y, or rotate180"
        )
    return np.ascontiguousarray(depth, dtype=np.float32)


def matrix_to_surface(
    matrix: np.ndarray, *, mapping: FixedMapping, finger: str, channel: str
) -> dict:
    """把一张 float32 深度贴到该指已经标定的表面采样上。不改网格。"""
    if finger not in mapping.fingers:
        raise ValueError(f"finger has no calibrated mapping: {finger!r}")
    calibration = mapping.fingers[finger]
    if channel != calibration["reference_channel"]:
        raise ValueError(
            f"channel {channel!r} does not match calibrated channel "
            f"{calibration['reference_channel']!r}"
        )
    values = mapping.apply(matrix, finger)
    valid = calibration["mapping_valid"] & np.isfinite(values)
    return {
        "vertices": mapping.vertices.copy(),
        "faces": mapping.faces.copy(),
        "xyz": calibration["target_xyz"].copy(),
        "values": values.copy(),
        "valid": valid,
        "face_id": calibration["face_id"].copy(),
        "barycentric": calibration["barycentric"].copy(),
        "source_pixel_indices": calibration["source_pixel_indices"].copy(),
    }


class _FingerUV:
    """一个手指的表面采样 → 官方 UV texel。未知样本保持 NaN。"""

    def __init__(
        self,
        mapping: FixedMapping,
        finger: str,
        *,
        low: float,
        high: float,
        side: str,
        size: int,
    ):
        if side not in ("left", "right"):
            raise ValueError("uv side must be left or right")
        if not isinstance(size, int) or size < 2:
            raise ValueError("UV size must be an integer >= 2")
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("depth range requires finite low < high")
        self.low, self.high = float(low), float(high)
        self.scale = (self.high - self.low) / 65534
        self.size, self.side, self.finger = size, side, finger
        record = mapping.fingers[finger]
        self.sample_count = len(record["face_id"])
        self.sample_indices = np.flatnonzero(record["mapping_valid"])
        texture, texture_faces = official_uv(
            side, np.ascontiguousarray(mapping.faces, dtype=np.int32).tobytes()
        )
        uv = np.einsum(
            "ni,nij->nj",
            record["barycentric"][self.sample_indices],
            texture[texture_faces[record["face_id"][self.sample_indices]]],
        )
        if not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv > 1):
            raise ValueError("Mapped UV coordinates are outside the official atlas")
        x = np.rint(uv[:, 0] * (size - 1)).astype(np.int64)
        y = np.rint((1 - uv[:, 1]) * (size - 1)).astype(np.int64)
        self.texels = y * size + x
        self.uv_sha256 = hashlib.sha256(
            (UV_DIR / f"MANO_UV_{side}.obj").read_bytes()
        ).hexdigest()

    def rasterize(self, values: np.ndarray, *, enforce_depth_range: bool) -> np.ndarray:
        values = np.asarray(values)
        if values.shape != (self.sample_count,):
            raise ValueError("Depth sample count does not match the fixed mapping")
        selected = values[self.sample_indices]
        finite = np.isfinite(selected)
        finite_values = selected[finite].astype(np.float64)
        if enforce_depth_range and (
            np.any(finite_values < self.low) or np.any(finite_values > self.high)
        ):
            raise ValueError("Depth outside PNG range; widen depth_min/depth_max")
        counts = np.bincount(self.texels[finite], minlength=self.size**2)
        sums = np.bincount(
            self.texels[finite], weights=finite_values, minlength=self.size**2
        )
        image = np.full(self.size**2, np.nan, dtype=np.float64)
        observed = counts > 0
        image[observed] = sums[observed] / counts[observed]
        return image.reshape(self.size, self.size)


class GripperHandUV:
    """两指 texel 均值等权合并，再编码接触代理 / 深度 / 有效 mask。"""

    def __init__(
        self,
        mapping: FixedMapping,
        *,
        side: str,
        size: int = 128,
        low: float = DEPTH_MIN,
        high: float = DEPTH_MAX,
        contact_min: float = CONTACT_MIN,
        contact_max: float = CONTACT_MAX,
        enforce_depth_range: bool = False,
        fingers: tuple[str, ...] = FINGERS,
    ):
        if (
            not np.isfinite([contact_min, contact_max]).all()
            or contact_max <= contact_min
        ):
            raise ValueError("Contact proxy requires finite contact_min < contact_max")
        self.mapping = mapping
        self.side, self.size = side, size
        self.low, self.high = float(low), float(high)
        self.contact_min, self.contact_max = float(contact_min), float(contact_max)
        self.enforce_depth_range = enforce_depth_range
        self.fingers = tuple(fingers)
        self.renderers = {
            finger: _FingerUV(
                mapping, finger, low=self.low, high=self.high, side=side, size=size
            )
            for finger in self.fingers
        }

    def rasterize(self, depths: dict[str, np.ndarray]) -> np.ndarray:
        if set(depths) != set(self.fingers):
            raise ValueError(f"depths must contain {list(self.fingers)}")
        total = np.zeros((self.size, self.size), np.float64)
        count = np.zeros(total.shape, np.int64)
        for finger in self.fingers:
            values = self.mapping.apply(depths[finger], finger)
            image = self.renderers[finger].rasterize(
                values, enforce_depth_range=self.enforce_depth_range
            )
            valid = np.isfinite(image)
            total[valid] += image[valid]
            count[valid] += 1
        observed = count > 0
        result = np.full(total.shape, np.nan, np.float64)
        result[observed] = total[observed] / count[observed]
        return result

    def outputs(self, frame_index: int, raster: np.ndarray, mode: str = "contact"):
        if mode not in ("depth", "contact", "both"):
            raise ValueError("mode must be depth, contact, or both")
        valid = np.isfinite(raster)
        values = raster[valid]
        if mode in ("depth", "both") and (
            np.any(values < self.low) or np.any(values > self.high)
        ):
            raise ValueError("Depth outside fixed UV encoding range")
        encoded = []
        stem = f"f{frame_index:04d}_{self.side}"
        if mode in ("depth", "both"):
            image = np.zeros(raster.shape, np.uint16)
            image[valid] = (
                np.rint((values - self.low) / (self.high - self.low) * 65534) + 1
            ).astype(np.uint16)
            encoded.append(encode_png(f"{stem}_depth.png", image))
        if mode in ("contact", "both"):
            image = np.zeros(raster.shape, np.uint8)
            image[valid] = np.rint(
                np.clip(
                    (values - self.contact_min) / (self.contact_max - self.contact_min),
                    0,
                    1,
                )
                * 255
            ).astype(np.uint8)
            encoded.append(encode_png(f"{stem}.png", image))
        mask = encode_png(f"{stem}_valid.png", valid.astype(np.uint8) * 255)
        return encoded, mask

    def preview(self, raster: np.ndarray) -> np.ndarray:
        """深度归一化后套 Sharpa 调色板，只作预览，不是接触图。"""
        valid = np.isfinite(raster)
        levels = np.zeros(raster.shape, np.float64)
        levels[valid] = (raster[valid] - self.low) / (self.high - self.low)
        rgb = np.empty((*raster.shape, 3), np.uint8)
        rgb[:] = BACKGROUND.astype(np.uint8)
        rgb[valid] = CONTACT_LUT[
            np.rint(np.clip(levels[valid], 0, 1) * 255).astype(np.uint8)
        ]
        return rgb


def retarget_frame(
    mapping: FixedMapping,
    depths: dict[str, np.ndarray],
    *,
    side: str,
    frame_index: int = 0,
    size: int = 128,
    mode: str = "contact",
    transforms: dict[str, str] | None = None,
    channels: dict[str, str] | None = None,
    depth_min: float = DEPTH_MIN,
    depth_max: float = DEPTH_MAX,
    contact_min: float = CONTACT_MIN,
    contact_max: float = CONTACT_MAX,
) -> dict:
    """一帧、一只手。``depths`` 的键是 ``thumb`` / ``index``。

    默认 ``mode="contact"``，与全量转换一致：接触代理加有效 mask，不写 uint16 深度。
    地图的 ``reference_channel`` 是左夹爪流名（拇指 ``left_tactile_left``，食指
    ``left_tactile_right``）。右手复用同一张地图时也对着这两个名字，不是右手流名。
    ``channels`` 只做这层核对；不传就用地图里的名字。
    """
    if side not in CHANNELS:
        raise ValueError("side must be left or right")
    if set(depths) != set(FINGERS):
        raise ValueError("depths must contain thumb and index")
    expected = channels or {
        finger: mapping.fingers[finger]["reference_channel"] for finger in FINGERS
    }
    if set(expected) != set(FINGERS):
        raise ValueError("channels must name thumb and index")
    for finger in FINGERS:
        calibrated = mapping.fingers[finger]["reference_channel"]
        if expected[finger] != calibrated:
            raise ValueError(
                f"{finger} channel {expected[finger]!r} does not match calibrated channel {calibrated!r}"
            )
    oriented = {}
    for finger, depth in depths.items():
        transform = (
            "identity" if transforms is None else transforms.get(finger, "identity")
        )
        oriented[finger] = transform_depth(depth, transform)
    renderer = GripperHandUV(
        mapping,
        side=side,
        size=size,
        low=depth_min,
        high=depth_max,
        contact_min=contact_min,
        contact_max=contact_max,
        enforce_depth_range=mode != "contact",
        fingers=FINGERS,
    )
    raster = renderer.rasterize(oriented)
    images, mask = renderer.outputs(frame_index, raster, mode)
    surfaces = {
        finger: matrix_to_surface(
            oriented[finger],
            mapping=mapping,
            finger=finger,
            channel=mapping.fingers[finger]["reference_channel"],
        )
        for finger in FINGERS
    }
    return {
        "raster": raster,
        "pngs": images,
        "mask": mask,
        "surfaces": surfaces,
        "side": side,
        "mode": mode,
    }
