import numpy as np


TABLE_TOP_Z = 0.0
TABLE_THICKNESS_M = 0.06
DEFAULT_SPHERE_RADIUS_M = 0.05


class MujocoSphereViewer:
    def __init__(self, radius_m=DEFAULT_SPHERE_RADIUS_M, max_spheres=1):
        self.radius_m = float(max(radius_m, 1e-4))
        self.max_spheres = max(1, int(max_spheres))
        self.mujoco = None
        self.viewer_module = None
        self.model = None
        self.data = None
        self.viewer = None
        self.geom_ids = []

    @property
    def available(self):
        return self.viewer is not None

    def start(self):
        try:
            import mujoco
            import mujoco.viewer
        except Exception as exc:
            print(f"MuJoCo unavailable: {exc}")
            print("Install it with: ../.venv/bin/python -m pip install mujoco")
            return False

        self.mujoco = mujoco
        self.viewer_module = mujoco.viewer
        xml = self._make_xml(self.radius_m)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"reconstructed_sphere_{idx}")
            for idx in range(self.max_spheres)
        ]
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        except Exception as exc:
            print(f"Could not launch MuJoCo viewer: {exc}")
            self.viewer = None
            return False
        print("MuJoCo sphere reconstruction viewer started.")
        return True

    def update(self, center_xyz, radius_m=None):
        self.update_spheres([{"center": center_xyz, "radius": radius_m if radius_m is not None else self.radius_m}])

    def update_spheres(self, spheres):
        if self.viewer is None:
            return

        with self.viewer.lock():
            for idx in range(self.max_spheres):
                qpos_start = 7 * idx
                geom_id = self.geom_ids[idx] if idx < len(self.geom_ids) else -1
                if idx < len(spheres):
                    center_xyz, radius_m = sphere_center_and_radius(spheres[idx], self.radius_m)
                    center = camera_y_up_to_mujoco_z_up(center_xyz)
                    radius = float(max(radius_m, 1e-4))
                    self.radius_m = radius
                    center[2] = max(center[2], TABLE_TOP_Z + radius)
                    self.data.qpos[qpos_start:qpos_start + 3] = center
                    self.data.qpos[qpos_start + 3:qpos_start + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                    if geom_id >= 0:
                        self.model.geom_size[geom_id, 0] = radius
                        self.model.geom_rgba[geom_id, 3] = 0.90
                else:
                    self.data.qpos[qpos_start:qpos_start + 3] = np.array([0.0, 0.0, -10.0], dtype=np.float64)
                    self.data.qpos[qpos_start + 3:qpos_start + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                    if geom_id >= 0:
                        self.model.geom_size[geom_id, 0] = 1e-4
                        self.model.geom_rgba[geom_id, 3] = 0.0
            self.data.qvel[:] = 0.0
            self.mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _make_xml(self, radius_m):
        radius_m = float(max(radius_m, 1e-4))
        table_half_x = max(0.75, 1.35 * radius_m)
        table_half_y = max(0.50, 1.05 * radius_m)
        camera_distance = max(1.4, 3.2 * radius_m)
        camera_height = max(0.75, 1.65 * radius_m)
        table_center_z = TABLE_TOP_Z - TABLE_THICKNESS_M * 0.5
        leg_half_height = 0.28
        leg_center_z = TABLE_TOP_Z - TABLE_THICKNESS_M - leg_half_height
        leg_x = max(0.08, table_half_x - 0.13)
        leg_y = max(0.08, table_half_y - 0.10)
        sphere_bodies = "\n".join(
            make_sphere_body_xml(idx, radius_m)
            for idx in range(self.max_spheres)
        )
        return f"""
<mujoco model="sphere_reconstruction">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <visual>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.65 0.65 0.65"/>
    <global azimuth="135" elevation="-25"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.20 0.22" rgb2="0.10 0.11 0.12" width="512" height="512"/>
    <material name="table_top" texture="grid" texrepeat="5 4" reflectance="0.08" rgba="0.50 0.46 0.38 1"/>
    <material name="table_leg" rgba="0.28 0.27 0.24 1"/>
    <material name="sphere_mat" rgba="0.10 0.75 0.25 0.90"/>
  </asset>
  <worldbody>
    <light pos="0 -3 4" dir="0 1 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="table_top" type="box" pos="0 0 {table_center_z:.4f}" size="{table_half_x:.4f} {table_half_y:.4f} {TABLE_THICKNESS_M * 0.5:.4f}" material="table_top"/>
    <geom name="table_leg_front_left" type="box" pos="-{leg_x:.4f} -{leg_y:.4f} {leg_center_z:.4f}" size="0.035 0.035 {leg_half_height:.4f}" material="table_leg"/>
    <geom name="table_leg_front_right" type="box" pos="{leg_x:.4f} -{leg_y:.4f} {leg_center_z:.4f}" size="0.035 0.035 {leg_half_height:.4f}" material="table_leg"/>
    <geom name="table_leg_back_left" type="box" pos="-{leg_x:.4f} {leg_y:.4f} {leg_center_z:.4f}" size="0.035 0.035 {leg_half_height:.4f}" material="table_leg"/>
    <geom name="table_leg_back_right" type="box" pos="{leg_x:.4f} {leg_y:.4f} {leg_center_z:.4f}" size="0.035 0.035 {leg_half_height:.4f}" material="table_leg"/>
    {sphere_bodies}
    <camera name="overview" pos="0 -{camera_distance:.4f} {camera_height:.4f}" xyaxes="1 0 0 0 0.45 0.89"/>
  </worldbody>
</mujoco>
"""


def make_sphere_body_xml(index, radius_m):
    radius_m = float(max(radius_m, 1e-4))
    x_offset = 0.10 * index
    return f"""
    <body name="sphere_body_{index}" pos="{x_offset:.4f} 0 {radius_m:.4f}">
      <freejoint name="sphere_freejoint_{index}"/>
      <geom name="reconstructed_sphere_{index}" type="sphere" size="{radius_m:.4f}" material="sphere_mat"/>
    </body>"""


def sphere_center_and_radius(sphere, fallback_radius):
    if isinstance(sphere, dict):
        center = sphere.get("center")
        radius = sphere.get("radius", fallback_radius)
    else:
        center, radius = sphere[:2]
    if radius is None or not np.isfinite(radius):
        radius = fallback_radius
    return center, float(radius)


def camera_y_up_to_mujoco_z_up(center_xyz):
    center = np.asarray(center_xyz, dtype=np.float64).reshape(3)
    # ZED camera coordinates here are treated as X-right, Y-up, Z-forward.
    # MuJoCo is Z-up, so forward depth becomes MuJoCo Y and camera-up becomes MuJoCo Z.
    return np.array([center[0], center[2], center[1]], dtype=np.float64)
