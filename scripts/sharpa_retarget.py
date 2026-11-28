"""Sharpa 单帧重定向。

22 自由度正运动学 → MANO 姿态（官方骨向量逆问题）→ 五指形变贴到指尖表面
→ 官方 MANO UV 上的 128×128 uint8 接触图。

灰度与生产一致：``clip(deform / 180, 0, 1)`` 经 inferno 调色板烘焙后反查回 uint8。
这是形变强度，不是物理力，也不是 TacVerse 的深度代理。缺测手指不补造触觉。

批量启动、GPU Jacobian、tar/NPZ 发布仍在 ``pipeline/sharpa_retarget`` 与
``scripts/sharpa/convert_sharpa.py``。本文件只放单帧算法，数值与那些模块一致。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from matplotlib import cm
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent
ASSETS = REPO / "assets"
URDF_DIR = ASSETS / "sharpa"
WAVE01 = ASSETS / "wave_01"
MANO_RIGHT = ASSETS / "mano" / "MANO_RIGHT.pkl"
UV_DIR = ASSETS / "mano" / "uv"

HANDS = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
STATE_GROUPS = {"left_hand": (7, 29), "right_hand": (36, 58)}

# URDF / 硬件顺序。Pinocchio 的 model.names 是树顺序，qpos 必须按关节名写入。
HAND_Q_NAMES = {
    "right": [
        "right_thumb_CMC_FE",
        "right_thumb_CMC_AA",
        "right_thumb_MCP_FE",
        "right_thumb_MCP_AA",
        "right_thumb_IP",
        "right_index_MCP_FE",
        "right_index_MCP_AA",
        "right_index_PIP",
        "right_index_DIP",
        "right_middle_MCP_FE",
        "right_middle_MCP_AA",
        "right_middle_PIP",
        "right_middle_DIP",
        "right_ring_MCP_FE",
        "right_ring_MCP_AA",
        "right_ring_PIP",
        "right_ring_DIP",
        "right_pinky_CMC",
        "right_pinky_MCP_FE",
        "right_pinky_MCP_AA",
        "right_pinky_PIP",
        "right_pinky_DIP",
    ],
    "left": [
        "left_thumb_CMC_FE",
        "left_thumb_CMC_AA",
        "left_thumb_MCP_FE",
        "left_thumb_MCP_AA",
        "left_thumb_IP",
        "left_index_MCP_FE",
        "left_index_MCP_AA",
        "left_index_PIP",
        "left_index_DIP",
        "left_middle_MCP_FE",
        "left_middle_MCP_AA",
        "left_middle_PIP",
        "left_middle_DIP",
        "left_ring_MCP_FE",
        "left_ring_MCP_AA",
        "left_ring_PIP",
        "left_ring_DIP",
        "left_pinky_CMC",
        "left_pinky_MCP_FE",
        "left_pinky_MCP_AA",
        "left_pinky_PIP",
        "left_pinky_DIP",
    ],
}

# MediaPipe 顺序：腕，拇指 1–4，食指 5–8，中指 9–12，无名指 13–16，小指 17–20。
MP_FRAME_SUFFIXES = (
    "hand_C_MC",
    "thumb_MC",
    "thumb_MCP_FE",
    "thumb_IP",
    "thumb_fingertip",
    "index_MCP_FE",
    "index_PIP",
    "index_DIP",
    "index_fingertip",
    "middle_MCP_FE",
    "middle_PIP",
    "middle_DIP",
    "middle_fingertip",
    "ring_MCP_FE",
    "ring_PIP",
    "ring_DIP",
    "ring_fingertip",
    "pinky_MCP_FE",
    "pinky_PIP",
    "pinky_DIP",
    "pinky_fingertip",
)

# manopth / dex-retargeting 右手指尖顶点，拇指到小指。
MANO_TIP_VERTS = (745, 317, 444, 556, 673)
TIP_VERT = {"thumb": 745, "index": 317, "middle": 444, "ring": 556, "pinky": 673}
DISTAL_JOINT = {"index": 3, "middle": 6, "pinky": 9, "ring": 12, "thumb": 15}
MANO_JOINT_TO_MP = (0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3)
MANO_TIP_TO_MP = (4, 8, 12, 16, 20)

# sharpa_wave_{left,right}.yml 的 origin→task 向量，这里反过来：机器人向量是目标。
VECTOR_ORIGIN_SUFFIX = (
    "hand_C_MC",
    "hand_C_MC",
    "hand_C_MC",
    "hand_C_MC",
    "hand_C_MC",
    "hand_C_MC",
    "thumb_MC",
    "index_PP",
    "middle_PP",
    "ring_PP",
    "pinky_PP",
    "thumb_DP",
    "index_DP",
    "middle_DP",
    "ring_DP",
    "pinky_DP",
)
VECTOR_TASK_SUFFIX = (
    "thumb_fingertip",
    "index_fingertip",
    "middle_fingertip",
    "ring_fingertip",
    "pinky_fingertip",
    "thumb_MC",
    "thumb_DP",
    "index_DP",
    "middle_DP",
    "ring_DP",
    "pinky_DP",
    "thumb_fingertip",
    "index_fingertip",
    "middle_fingertip",
    "ring_fingertip",
    "pinky_fingertip",
)
HUMAN_ORIGIN_IDX = (0, 0, 0, 0, 0, 0, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19)
HUMAN_TASK_IDX = (4, 8, 12, 16, 20, 2, 3, 7, 11, 15, 19, 4, 8, 12, 16, 20)

SKIN = np.array([210.0, 168.0, 140.0], dtype=np.float32)
BACKGROUND = np.array([24, 24, 28], dtype=np.int16)


# ---------------------------------------------------------------------------
# MANO
# ---------------------------------------------------------------------------


class _ChumpyDummy:
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _ChumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _ChumpyDummy
        return super().find_class(module, name)


def load_mano_pkl(path: Path | None = None) -> dict:
    path = Path(path or MANO_RIGHT)
    with path.open("rb") as handle:
        raw = _ChumpyUnpickler(handle, encoding="latin1").load()
    return {
        "v_template": np.asarray(raw["v_template"], dtype=np.float64),
        "f": np.asarray(raw["f"], dtype=np.int32),
        "weights": np.asarray(raw["weights"], dtype=np.float64),
        "J_regressor": raw["J_regressor"].toarray().astype(np.float64),
        "posedirs": np.asarray(raw["posedirs"], dtype=np.float64),
        "hands_mean": np.asarray(raw["hands_mean"], dtype=np.float64),
        "hands_components": np.asarray(raw["hands_components"], dtype=np.float64),
        "kintree_table": np.asarray(raw["kintree_table"]),
        "J": np.asarray(raw["J"], dtype=np.float64),
    }


def _rodrigues(rotation: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotation, axis=1, keepdims=True)
    small = theta[:, 0] < 1e-8
    theta_safe = np.clip(theta, 1e-8, None)
    direction = rotation / theta_safe
    nx, ny, nz = direction[:, 0], direction[:, 1], direction[:, 2]
    skew = np.zeros((len(rotation), 3, 3), dtype=np.float64)
    skew[:, 0, 1], skew[:, 0, 2] = -nz, ny
    skew[:, 1, 0], skew[:, 1, 2] = nz, -nx
    skew[:, 2, 0], skew[:, 2, 1] = -ny, nx
    eye = np.eye(3)[None]
    angle = theta_safe[:, 0]
    matrix = (
        eye
        + np.sin(angle)[:, None, None] * skew
        + (1.0 - np.cos(angle))[:, None, None] * (skew @ skew)
    )
    if small.any():
        matrix[small] = eye[0]
    return matrix


def mano_forward(
    model: dict,
    pca: np.ndarray | None = None,
    ncomps: int = 15,
    flat_hand_mean: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """MANO 顶点与 16 个关节，单位毫米。``flat_hand_mean`` 跳过 hands_mean。"""
    pca = (
        np.zeros(ncomps, dtype=np.float64)
        if pca is None
        else np.asarray(pca, dtype=np.float64)
    )
    pose45 = pca @ model["hands_components"][:ncomps]
    if not flat_hand_mean:
        pose45 = model["hands_mean"] + pose45
    relative = _rodrigues(pose45.reshape(15, 3))
    pose_feature = (relative - np.eye(3)[None]).reshape(-1)
    vertices = (
        model["v_template"] + model["posedirs"].reshape(778, 3, 135) @ pose_feature
    )
    joints = model["J_regressor"] @ vertices

    parent = {i: int(model["kintree_table"][0, i]) for i in range(16)}
    rotation = np.concatenate([np.eye(3)[None], relative], axis=0)
    posed = np.zeros((16, 4, 4), dtype=np.float64)
    posed[0, :3, :3] = rotation[0]
    posed[0, :3, 3] = joints[0]
    posed[0, 3, 3] = 1.0
    for index in range(1, 16):
        local = np.eye(4)
        local[:3, :3] = rotation[index]
        local[:3, 3] = joints[index] - joints[parent[index]]
        posed[index] = posed[parent[index]] @ local

    bone = posed.copy()
    bone[:, :3, 3] -= np.einsum("nij,nj->ni", posed[:, :3, :3], joints)
    transforms = np.einsum("vj,jab->vab", model["weights"], bone)
    homogeneous = np.concatenate([vertices, np.ones((778, 1))], axis=1)
    posed_vertices = np.einsum("vab,vb->va", transforms, homogeneous)[:, :3]
    return (posed_vertices * 1000.0).astype(np.float32), (
        posed[:, :3, 3] * 1000.0
    ).astype(np.float32)


def mano_mp21(vertices: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """MANO 关节加指尖，排成 MediaPipe 21 点，单位毫米。"""
    keypoints = np.zeros((21, 3), dtype=np.float64)
    keypoints[0] = joints[0]
    keypoints[1], keypoints[2], keypoints[3], keypoints[4] = (
        joints[13],
        joints[14],
        joints[15],
        vertices[MANO_TIP_VERTS[0]],
    )
    keypoints[5], keypoints[6], keypoints[7], keypoints[8] = (
        joints[1],
        joints[2],
        joints[3],
        vertices[MANO_TIP_VERTS[1]],
    )
    keypoints[9], keypoints[10], keypoints[11], keypoints[12] = (
        joints[4],
        joints[5],
        joints[6],
        vertices[MANO_TIP_VERTS[2]],
    )
    keypoints[13], keypoints[14], keypoints[15], keypoints[16] = (
        joints[10],
        joints[11],
        joints[12],
        vertices[MANO_TIP_VERTS[3]],
    )
    keypoints[17], keypoints[18], keypoints[19], keypoints[20] = (
        joints[7],
        joints[8],
        joints[9],
        vertices[MANO_TIP_VERTS[4]],
    )
    return keypoints


# ---------------------------------------------------------------------------
# Sharpa FK 与姿态拟合
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def load_sharpa_hand(side: str, urdf_dir: str | None = None):
    import pinocchio as pin

    root = Path(urdf_dir) if urdf_dir else URDF_DIR
    model = pin.buildModelFromUrdf(str(root / f"{side}_sharpa_wave_with_wrist.urdf"))
    return model, model.createData()


def qpos_from_hand_state(model, side: str, hand_q: np.ndarray) -> np.ndarray:
    import pinocchio as pin

    configuration = pin.neutral(model)
    if len(hand_q) != 22:
        raise ValueError(f"expected 22-DoF hand, got {len(hand_q)}")
    for name, value in zip(HAND_Q_NAMES[side], hand_q):
        joint = model.joints[model.getJointId(name)]
        configuration[joint.idx_q] = float(value)
    return configuration


def skeleton_xyz_m(
    side: str, hand_q: np.ndarray, center_wrist: bool = True
) -> np.ndarray:
    """21 个 MediaPipe 顺序的点，单位米，默认把腕关节平移到原点。"""
    import pinocchio as pin

    model, data = load_sharpa_hand(side)
    configuration = qpos_from_hand_state(model, side, hand_q)
    pin.forwardKinematics(model, data, configuration)
    pin.updateFramePlacements(model, data)
    points = np.zeros((21, 3), dtype=np.float64)
    for index, suffix in enumerate(MP_FRAME_SUFFIXES):
        frame = model.getFrameId(f"{side}_{suffix}")
        points[index] = np.asarray(data.oMf[frame].translation)
    if center_wrist:
        points = points - points[0]
    return points


def robot_vector_xyz_mm(side: str, hand_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """官方 YAML 的 16 对 origin/task 点，单位毫米。"""
    import pinocchio as pin

    model, data = load_sharpa_hand(side)
    configuration = qpos_from_hand_state(model, side, hand_q)
    pin.forwardKinematics(model, data, configuration)
    pin.updateFramePlacements(model, data)
    origin = np.zeros((16, 3), dtype=np.float64)
    task = np.zeros((16, 3), dtype=np.float64)
    for index, (origin_suffix, task_suffix) in enumerate(
        zip(VECTOR_ORIGIN_SUFFIX, VECTOR_TASK_SUFFIX)
    ):
        origin[index] = (
            np.asarray(
                data.oMf[model.getFrameId(f"{side}_{origin_suffix}")].translation
            )
            * 1000.0
        )
        task[index] = (
            np.asarray(data.oMf[model.getFrameId(f"{side}_{task_suffix}")].translation)
            * 1000.0
        )
    return origin, task


def _umeyama(
    source: np.ndarray, destination: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """相似变换 destination ≈ scale * R @ source + translation。"""
    count = source.shape[0]
    source_mean, destination_mean = source.mean(0), destination.mean(0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean
    covariance = (destination_centered.T @ source_centered) / count
    left, singular, right = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(left) * np.linalg.det(right) < 0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right
    variance = float((source_centered**2).sum() / count)
    scale = float(np.sum(singular * sign) / max(variance, 1e-12))
    translation = destination_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _rodrigues_vec(rotation: np.ndarray) -> np.ndarray:
    return _rodrigues(np.asarray(rotation, dtype=np.float64).reshape(1, 3))[0]


def fit_mano_to_skeleton(
    mano_model: dict,
    skeleton_mm: np.ndarray,
    ncomps: int = 15,
    left: bool = False,
    robot_vectors: tuple[np.ndarray, np.ndarray] | None = None,
    x0: np.ndarray | None = None,
    max_nfev: int = 400,
) -> dict:
    """拟合腕旋转 + 15 维 PCA，使 MANO 骨向量对齐 Sharpa，再用 Umeyama 坐下网格。

    函数默认最多 400 次评估。逐帧生产循环传入 ``max_nfev=40``，并用上一帧的 ``x`` 暖启动。
    """
    from scipy.optimize import least_squares

    target = np.asarray(skeleton_mm, dtype=np.float64)
    origin_index = np.array(HUMAN_ORIGIN_IDX, dtype=int)
    task_index = np.array(HUMAN_TASK_IDX, dtype=int)
    tip_ids = np.array(MANO_TIP_VERTS)
    tip_map = np.array(MANO_TIP_TO_MP)
    joint_map = np.array(MANO_JOINT_TO_MP)
    if robot_vectors is None:
        robot = target[task_index] - target[origin_index]
    else:
        origin, task = robot_vectors
        robot = task - origin
    robot_unit = robot / np.clip(
        np.linalg.norm(robot, axis=1, keepdims=True), 1e-8, None
    )

    def posed(pca):
        vertices, joints = mano_forward(
            mano_model, pca=pca, ncomps=ncomps, flat_hand_mean=True
        )
        vertices = vertices.astype(np.float64)
        joints = joints.astype(np.float64)
        if left:
            vertices[:, 0] *= -1.0
            joints[:, 0] *= -1.0
        return vertices, joints

    def pack_pose(parameters):
        vertices, joints = posed(parameters[3:])
        rotation = _rodrigues_vec(parameters[:3])
        return (rotation @ vertices.T).T, (rotation @ joints.T).T

    def residual(parameters):
        vertices, joints = pack_pose(parameters)
        keypoints = mano_mp21(vertices, joints)
        bone = keypoints[task_index] - keypoints[origin_index]
        unit = bone / np.clip(np.linalg.norm(bone, axis=1, keepdims=True), 1e-8, None)
        return np.concatenate([(unit - robot_unit).ravel(), 0.08 * parameters[3:]])

    if x0 is None:
        initial = np.zeros(3 + ncomps, dtype=np.float64)
    else:
        initial = np.asarray(x0, dtype=np.float64).reshape(-1)
        if initial.size != 3 + ncomps:
            initial = np.zeros(3 + ncomps, dtype=np.float64)
    result = least_squares(
        residual, initial, method="trf", max_nfev=max_nfev, ftol=1e-5, xtol=1e-5
    )
    vertices, joints = pack_pose(result.x)
    keypoints = mano_mp21(vertices, joints)
    rigid = np.array([0, 1, 5, 9, 13, 17, 4, 8, 12, 16, 20], dtype=int)
    scale, rotation, translation = _umeyama(keypoints[rigid], target[rigid])
    vertices = scale * (rotation @ vertices.T).T + translation
    joints = scale * (rotation @ joints.T).T + translation
    tips = vertices[tip_ids]
    tip_error = np.linalg.norm(tips - target[tip_map], axis=1)
    joint_error = np.linalg.norm(joints - target[joint_map], axis=1)
    seated = mano_mp21(vertices, joints)
    seated_bone = seated[task_index] - seated[origin_index]
    seated_unit = seated_bone / np.clip(
        np.linalg.norm(seated_bone, axis=1, keepdims=True), 1e-8, None
    )
    angle = np.degrees(np.arccos(np.clip((seated_unit * robot_unit).sum(1), -1.0, 1.0)))
    return {
        "verts": vertices.astype(np.float32),
        "joints": joints.astype(np.float32),
        "tips": tips,
        "target": target[tip_map],
        "skel": target,
        "err_mm": tip_error,
        "err_joint_mm": joint_error,
        "rms_mm": float(np.sqrt((tip_error**2).mean())),
        "rms_joint_mm": float(np.sqrt((joint_error**2).mean())),
        "scale": float(scale),
        "vec_deg": angle,
        "vec_deg_mean": float(angle.mean()),
        "x": result.x,
        "nfev": int(result.nfev),
        "rotation": rotation @ _rodrigues_vec(result.x[:3]),
        "translation_mm": translation,
        "solver_status": int(result.status),
        "solver_success": bool(result.success),
    }


# ---------------------------------------------------------------------------
# 五指形变贴到 MANO 指尖
# ---------------------------------------------------------------------------


@dataclass
class SensorPoints:
    name: str
    points: np.ndarray
    valid: np.ndarray


def load_sensor_assets(asset_dir: Path | None = None) -> dict[str, SensorPoints]:
    asset_dir = Path(asset_dir or WAVE01)
    assets = {}
    for key, stem in (("thumb", "TH"), ("finger", "4F")):
        points = np.load(
            asset_dir / f"tactileSensor_map_{stem}_point.npy", allow_pickle=False
        )
        if points.shape != (240, 240, 3) or not np.isfinite(points).all():
            raise ValueError("expected finite factory point map [240, 240, 3]")
        assets[key] = SensorPoints(key, points, np.linalg.norm(points, axis=-1) > 1e-4)
    return assets


def sensor_for_finger(assets: dict[str, SensorPoints], finger: str) -> SensorPoints:
    return assets["thumb"] if finger == "thumb" else assets["finger"]


def deform_luma(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return frame.astype(np.float32)


def resize_deform(gray: np.ndarray, hw: tuple[int, int] = (240, 240)) -> np.ndarray:
    if gray.shape[:2] == hw:
        return gray
    return cv2.resize(gray, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


def records_from_frames(assets: dict[str, SensorPoints], frames: dict) -> list[dict]:
    """``frames`` 的键是 ``left_thumb`` 这种 ``{hand}_{finger}``。缺键表示该路没测到。"""
    mapped = []
    for hand in HANDS:
        for finger in FINGERS:
            key = f"{hand}_{finger}"
            gray = frames.get(key)
            mesh = sensor_for_finger(assets, finger)
            if gray is None:
                mapped.append({"hand": hand, "finger": finger, "ok": False})
                continue
            gray = resize_deform(deform_luma(np.asarray(gray)))
            mapped.append(
                {
                    "hand": hand,
                    "finger": finger,
                    "ok": True,
                    "key": key,
                    "gray": gray,
                    "mesh": mesh.name,
                    "peak": float(gray.max()) if gray.size else 0.0,
                    "mean": float(gray.mean()) if gray.size else 0.0,
                }
            )
    return mapped


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-8 else vector


@dataclass
class FingerFrame:
    origin: np.ndarray
    rot: np.ndarray
    scale: float
    vert_ids: np.ndarray


def distal_frame(
    vertices: np.ndarray, joints: np.ndarray, weights: np.ndarray, finger: str
) -> FingerFrame:
    joint_id = DISTAL_JOINT[finger]
    tip = vertices[TIP_VERT[finger]]
    toward_tip = _normalize(tip - joints[joint_id])
    palmar_guess = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    width = np.cross(toward_tip, palmar_guess)
    if np.linalg.norm(width) < 0.15:
        palmar_guess = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        width = np.cross(toward_tip, palmar_guess)
    width = _normalize(width)
    palmar = _normalize(np.cross(width, toward_tip))
    rotation = np.stack([width, -toward_tip, palmar], axis=1)
    ids = np.where(weights[:, joint_id] > 0.2)[0]
    if TIP_VERT[finger] not in ids:
        ids = np.unique(np.append(ids, TIP_VERT[finger]))
    bone_length = float(np.linalg.norm(tip - joints[joint_id]))
    return FingerFrame(
        origin=tip.astype(np.float64), rot=rotation, scale=bone_length, vert_ids=ids
    )


_SENSOR_TREE: dict[str, tuple] = {}


def _sensor_tree(mesh: SensorPoints):
    cached = _SENSOR_TREE.get(mesh.name)
    if cached is not None:
        return cached
    pixels = np.argwhere(mesh.valid)
    source = np.asarray(mesh.points[mesh.valid], dtype=np.float64)
    y_tip = float(source[:, 1].min()) if len(source) else 0.0
    local = np.stack([source[:, 0], source[:, 1] - y_tip, -source[:, 2]], axis=1)
    cached = (cKDTree(local), pixels, y_tip)
    _SENSOR_TREE[mesh.name] = cached
    return cached


def paint_mano(
    mano_vertices: np.ndarray,
    mano_faces: np.ndarray,
    weights: np.ndarray,
    joints: np.ndarray,
    assets: dict[str, SensorPoints],
    mapped: list[dict],
    hand: str,
    max_dist_mm: float = 6.5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """每个指尖顶点取最近的传感器像素。6.5 mm 内才写入形变，更近的手指优先。"""
    del mano_faces
    values = np.zeros(len(mano_vertices), dtype=np.float32)
    hits = np.zeros(len(mano_vertices), dtype=bool)
    finger_id = np.full(len(mano_vertices), -1, dtype=np.int32)
    sensor_uv = np.zeros((len(mano_vertices), 2), dtype=np.float32)
    distance_out = np.full(len(mano_vertices), np.inf, dtype=np.float32)
    records = {
        f"{item['hand']}_{item['finger']}": item for item in mapped if item.get("ok")
    }
    for finger_index, finger in enumerate(FINGERS):
        record = records.get(f"{hand}_{finger}")
        if record is None:
            continue
        mesh = sensor_for_finger(assets, finger)
        frame = distal_frame(mano_vertices, joints, weights, finger)
        tree, pixels, y_tip = _sensor_tree(mesh)
        if len(pixels) == 0:
            continue
        gray = record["gray"][pixels[:, 0], pixels[:, 1]].astype(np.float32)
        scale = frame.scale / max(abs(y_tip) * 1.1, 1.0)
        ids = frame.vert_ids
        query = np.asarray(mano_vertices[ids], dtype=np.float64)
        local_query = (frame.rot.T @ (query - frame.origin).T).T / scale
        local_distance, nearest = tree.query(local_query)
        distance = local_distance * scale
        better = distance < distance_out[ids]
        if not np.any(better):
            continue
        height, width = record["gray"].shape[:2]
        rows = pixels[nearest, 0]
        cols = pixels[nearest, 1]
        selected = ids[better]
        distance_out[selected] = distance[better]
        sensor_uv[selected, 0] = (cols[better].astype(np.float32) + 0.5) / float(width)
        sensor_uv[selected, 1] = (rows[better].astype(np.float32) + 0.5) / float(height)
        finger_id[selected] = finger_index
        paint = better & (distance <= max_dist_mm)
        clear = better & ~paint
        values[ids[paint]] = gray[nearest[paint]]
        hits[ids[paint]] = True
        values[ids[clear]] = 0.0
        hits[ids[clear]] = False
    level = np.clip(values / 180.0, 0.0, 1.0)
    hot = (cm.get_cmap("inferno")(level)[..., :3] * 255.0).astype(np.float32)
    weight = np.clip(level[..., None] * 2.2, 0.0, 1.0)
    rgb = SKIN * (1.0 - weight) + hot * weight
    rgb[~hits] = SKIN
    assignment = {"finger": finger_id, "uv": sensor_uv, "dist": distance_out}
    return np.clip(rgb, 0, 255).astype(np.uint8), values, assignment


# ---------------------------------------------------------------------------
# 官方 MANO UV → 128 灰度接触图
# ---------------------------------------------------------------------------


def _contact_lut() -> np.ndarray:
    level = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    hot = cm.get_cmap("inferno")(level)[..., :3] * 255.0
    weight = np.clip(level * 2.2, 0.0, 1.0)[..., None]
    mixed = SKIN.astype(np.float32) * (1.0 - weight) + hot * weight
    return np.clip(np.rint(mixed), 0, 255).astype(np.int16)


CONTACT_LUT = _contact_lut()


def rgb_uv_to_contact(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected [H, W, 3] RGB, received {rgb.shape}")
    flat = np.ascontiguousarray(rgb.reshape(-1, 3))
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    colors = unique.astype(np.int16)
    background = np.abs(colors - BACKGROUND).sum(axis=1) <= 18
    delta = colors[:, None, :] - CONTACT_LUT[None, :, :]
    distance = np.square(delta).sum(axis=2)
    nearest = distance.argmin(axis=1)
    residual = distance[np.arange(colors.shape[0]), nearest]
    gray = nearest.astype(np.uint8)
    gray[background | (residual > 35 * 35)] = 0
    return gray[inverse].reshape(rgb.shape[:2])


def _parse_uv_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    texture, faces, texture_faces = [], [], []
    for line in path.read_text().splitlines():
        if line.startswith("vt "):
            texture.append([float(value) for value in line.split()[1:3]])
        elif line.startswith("f "):
            face, texture_face = [], []
            for token in line.split()[1:]:
                parts = token.split("/")
                face.append(int(parts[0]) - 1)
                texture_face.append(
                    int(parts[1]) - 1
                    if len(parts) > 1 and parts[1]
                    else int(parts[0]) - 1
                )
            faces.append(face[:3])
            texture_faces.append(texture_face[:3])
    return (
        np.asarray(texture, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
        np.asarray(texture_faces, dtype=np.int32),
    )


def _align_ft(
    source_faces: np.ndarray, source_ft: np.ndarray, destination_faces: np.ndarray
) -> np.ndarray:
    if np.array_equal(source_faces, destination_faces):
        return source_ft
    lookup = {
        tuple(sorted(int(vertex) for vertex in triangle)): index
        for index, triangle in enumerate(source_faces)
    }
    aligned = np.empty_like(source_ft)
    for index, triangle in enumerate(destination_faces):
        source_index = lookup[tuple(sorted(int(vertex) for vertex in triangle))]
        source = source_faces[source_index]
        order = [int(np.where(source == int(vertex))[0][0]) for vertex in triangle]
        aligned[index] = source_ft[source_index, order]
    return aligned


@lru_cache(maxsize=4)
def official_uv(side: str, faces_key: bytes) -> tuple[np.ndarray, np.ndarray]:
    faces = np.frombuffer(faces_key, dtype=np.int32).reshape(-1, 3)
    texture, source_faces, source_ft = _parse_uv_obj(UV_DIR / f"MANO_UV_{side}.obj")
    return texture, _align_ft(source_faces, source_ft, faces)


def _shift(source: np.ndarray, dy: int, dx: int) -> np.ndarray:
    shifted = np.full_like(source, -1)
    height = source.shape[0] - abs(dy)
    width = source.shape[1] - abs(dx)
    y_source, y_destination = max(dy, 0), max(-dy, 0)
    x_source, x_destination = max(dx, 0), max(-dx, 0)
    shifted[
        y_destination : y_destination + height, x_destination : x_destination + width
    ] = source[y_source : y_source + height, x_source : x_source + width]
    return shifted


def _label_map(corner_uv: np.ndarray, size: int = 2048) -> np.ndarray:
    raw = np.full((size, size), np.uint16(65535))
    for face_index in range(len(corner_uv)):
        px = np.round(corner_uv[face_index, :, 0] * (size - 1)).astype(np.int32)
        py = np.round((1.0 - corner_uv[face_index, :, 1]) * (size - 1)).astype(np.int32)
        cv2.fillConvexPoly(raw, np.stack([px, py], axis=1), int(face_index))
    label = raw.astype(np.int32)
    label[raw == np.uint16(65535)] = -1
    mask = (label >= 0).astype(np.uint8)
    grown = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    for dy, dx in (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ):
        missing = (grown > 0) & (label < 0)
        if not np.any(missing):
            break
        shifted = _shift(label, dy, dx)
        take = missing & (shifted >= 0)
        label[take] = shifted[take]
    return label


_LABELS: dict[tuple, np.ndarray] = {}


def texture_from_rgb(
    side: str, faces: np.ndarray, face_rgb: np.ndarray, size: int = 1024
) -> np.ndarray:
    key = (side, size)
    label = _LABELS.get(key)
    if label is None:
        texture, texture_faces = official_uv(
            side, np.ascontiguousarray(faces, dtype=np.int32).tobytes()
        )
        label = _label_map(texture[texture_faces], size)
        _LABELS[key] = label
    image = np.empty((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    image[:] = (24, 24, 28)
    selected = label >= 0
    image[selected] = np.clip(face_rgb[label[selected]], 0, 255).astype(np.uint8)
    return image


def encode_png(name: str, pixels: np.ndarray) -> tuple[str, bytes]:
    image = pixels[:, :, ::-1] if pixels.ndim == 3 else pixels
    ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError(f"PNG encode failed: {name}")
    return name, data.tobytes()


def encode_gray_uv(
    frame_index: int, hands: dict[str, dict], gray_size: int = 128
) -> list[tuple[str, bytes]]:
    """先在 1024 官方 UV 上铺面色，再最近邻缩到 128，最后反查接触灰度。"""
    encoded = []
    for side, spec in hands.items():
        faces = np.asarray(spec["faces"], dtype=np.int32)
        rgb = np.asarray(spec["rgb"], dtype=np.float32)
        texture = texture_from_rgb(side, faces, rgb[faces].mean(axis=1))
        if texture.shape[:2] != (gray_size, gray_size):
            texture = cv2.resize(
                texture, (gray_size, gray_size), interpolation=cv2.INTER_NEAREST
            )
        gray = rgb_uv_to_contact(texture)
        encoded.append(encode_png(f"f{frame_index:04d}_{side}.png", gray))
    return encoded


def hand_q_from_state(state: np.ndarray, side: str) -> np.ndarray:
    start, stop = STATE_GROUPS[f"{side}_hand"]
    return np.asarray(state, dtype=np.float64)[start:stop]


def retarget_frame(
    mano: dict,
    assets: dict[str, SensorPoints],
    q_left: np.ndarray,
    q_right: np.ndarray,
    frames: dict,
    *,
    ncomps: int = 15,
    x_left: np.ndarray | None = None,
    x_right: np.ndarray | None = None,
    frame_index: int = 0,
    gray_size: int = 128,
    max_nfev: int = 40,
) -> dict:
    """一帧双手。把返回的 ``x_left`` / ``x_right`` 传给下一帧做暖启动。

    ``frames`` 只放测到的路，例如 ``{"right_index": gray_hw}``。
    ``pngs`` 是 ``(f0000_left.png, bytes)`` 与右手对应项。
    """
    q_left = np.asarray(q_left, dtype=np.float64)
    q_right = np.asarray(q_right, dtype=np.float64)
    mapped = records_from_frames(assets, frames)
    faces_right = mano["f"]
    faces_left = faces_right[:, [0, 2, 1]]
    fit_left = fit_mano_to_skeleton(
        mano,
        skeleton_xyz_m("left", q_left, center_wrist=True) * 1000.0,
        ncomps=ncomps,
        left=True,
        robot_vectors=robot_vector_xyz_mm("left", q_left),
        x0=x_left,
        max_nfev=max_nfev,
    )
    fit_right = fit_mano_to_skeleton(
        mano,
        skeleton_xyz_m("right", q_right, center_wrist=True) * 1000.0,
        ncomps=ncomps,
        left=False,
        robot_vectors=robot_vector_xyz_mm("right", q_right),
        x0=x_right,
        max_nfev=max_nfev,
    )
    color_left, values_left, uv_left = paint_mano(
        fit_left["verts"],
        faces_left,
        mano["weights"],
        fit_left["joints"],
        assets,
        mapped,
        "left",
    )
    color_right, values_right, uv_right = paint_mano(
        fit_right["verts"],
        faces_right,
        mano["weights"],
        fit_right["joints"],
        assets,
        mapped,
        "right",
    )
    pngs = encode_gray_uv(
        frame_index,
        {
            "left": {"faces": faces_left, "rgb": color_left},
            "right": {"faces": faces_right, "rgb": color_right},
        },
        gray_size=gray_size,
    )
    return {
        "fits": {"left": fit_left, "right": fit_right},
        "vertex_rgb": {"left": color_left, "right": color_right},
        "vertex_values": {"left": values_left, "right": values_right},
        "sensor_uv": {"left": uv_left, "right": uv_right},
        "pngs": pngs,
        "x_left": fit_left["x"],
        "x_right": fit_right["x"],
    }


def load_runtime(mano_pkl: Path | None = None, wave_dir: Path | None = None):
    return load_mano_pkl(mano_pkl), load_sensor_assets(wave_dir)
