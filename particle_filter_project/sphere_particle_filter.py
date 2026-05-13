from dataclasses import dataclass

import numpy as np

from sphere_detection import SphereEstimate3D


@dataclass
class ParticleFilterConfig:
    particle_count: int = 400
    initial_position_std_m: float = 0.035
    process_position_std_m: float = 0.018
    measurement_position_std_m: float = 0.025
    measurement_radius_std_m: float = 0.004
    lock_radius: bool = True
    fixed_radius_m: float = 0.0
    radius_calibration_frames: int = 1
    radius_calibration_min_frames: int = 1
    radius_calibration_relative_mad: float = 0.15
    surface_likelihood: bool = True
    surface_likelihood_points: int = 192
    surface_distance_std_m: float = 0.035
    surface_likelihood_weight: float = 0.70
    surface_pseudo_count: float = 8.0
    surface_trim_fraction: float = 0.70
    surface_position_blend_scale: float = 0.65
    min_surface_likelihood_points: int = 12
    surface_normal_likelihood: bool = False
    surface_normal_std: float = 0.25
    surface_normal_weight: float = 0.0
    surface_area_weighting: bool = True
    projection_likelihood: bool = True
    projection_center_std_px: float = 16.0
    projection_radius_std_px: float = 24.0
    projection_likelihood_weight: float = 0.65
    adaptive_noise: bool = True
    residual_noise_gain: float = 6.0
    lost_noise_gain: float = 0.30
    max_noise_scale: float = 4.0
    reinitialize_distance_m: float = 0.220
    position_measurement_blend: float = 0.35
    measurement_proposal: bool = True
    measurement_proposal_ratio: float = 0.45
    measurement_proposal_min_particles: int = 48
    proposal_lateral_std_m: float = 0.018
    proposal_depth_std_m: float = 0.040
    resample_effective_ratio: float = 0.50
    max_prediction_frames: int = 15
    min_radius_m: float = 0.005
    max_radius_m: float = 1.0


@dataclass
class ParticleFilterResult:
    center_xyz: np.ndarray
    velocity_xyz: np.ndarray
    radius_m: float
    effective_sample_size: float
    measurement_used: bool
    prediction_only: bool
    lost_frames: int
    model_likelihood_used: bool
    projection_likelihood_used: bool
    measurement_proposal_used: bool
    radius_calibrated: bool
    radius_sample_count: int
    motion_noise_scale: float


class SphereParticleFilter:
    """Position-only particle filter for a fixed-size ball target."""

    def __init__(self, config=None, seed=13):
        self.config = config if config is not None else ParticleFilterConfig()
        self.rng = np.random.default_rng(seed)
        self.particles = None
        self.weights = None
        self.initialized = False
        self.lost_frames = 0
        self.radius_m = self._configured_fixed_radius()
        self.radius_calibrated = self.radius_m is not None
        self.radius_samples = []
        self.last_model_likelihood_used = False
        self.last_projection_likelihood_used = False
        self.last_measurement_proposal_used = False
        self.last_motion_noise_scale = 1.0

    def step(self, measurement, dt, camera_intrinsics=None):
        dt = float(np.clip(dt, 1e-3, 0.20))
        if measurement is not None and not self._valid_measurement(measurement):
            measurement = None

        if measurement is not None:
            self._update_radius_from_measurement(measurement)
            self.last_model_likelihood_used = False
            self.last_projection_likelihood_used = False
            self.last_measurement_proposal_used = False
            self.last_motion_noise_scale = self._adaptive_noise_scale(measurement)

            if not self.initialized or self._should_reinitialize(measurement):
                self._initialize(measurement)
                self.last_model_likelihood_used = self._has_surface_model_measurement(measurement)
                self.last_projection_likelihood_used = self._has_projection_measurement(measurement, camera_intrinsics)
            else:
                self._predict(dt, self.last_motion_noise_scale)
                self._inject_measurement_proposal(measurement, camera_intrinsics=camera_intrinsics)
                self._weight(measurement, camera_intrinsics=camera_intrinsics)
                self._maybe_resample(noise_scale=self.last_motion_noise_scale)
                self._blend_position_measurement(measurement)

            self.lost_frames = 0
            return self._estimate(measurement_used=True, prediction_only=False)

        if not self.initialized:
            return None

        self.lost_frames += 1
        self.last_model_likelihood_used = False
        self.last_projection_likelihood_used = False
        self.last_measurement_proposal_used = False
        self.last_motion_noise_scale = self._lost_noise_scale()
        if self.lost_frames > int(self.config.max_prediction_frames):
            self.initialized = False
            return None

        self._predict(dt, self.last_motion_noise_scale)
        self._maybe_resample(force=self.lost_frames > 1, noise_scale=self.last_motion_noise_scale)
        return self._estimate(measurement_used=False, prediction_only=True)

    def _initialize(self, measurement):
        count = max(32, int(self.config.particle_count))
        center = np.asarray(measurement.center_xyz, dtype=np.float64).reshape(3)

        self.particles = center[None, :] + self.rng.normal(0.0, self.config.initial_position_std_m, (count, 3))
        self.weights = np.full(count, 1.0 / count, dtype=np.float64)
        self.initialized = True

    def _predict(self, dt, noise_scale=1.0):
        if self.particles is None:
            return

        count = len(self.particles)
        noise_scale = float(np.clip(noise_scale, 0.25, self.config.max_noise_scale))
        self.particles += self.rng.normal(
            0.0,
            self.config.process_position_std_m * noise_scale,
            (count, 3),
        )

    def _inject_measurement_proposal(self, measurement, camera_intrinsics=None):
        if not bool(self.config.measurement_proposal):
            return
        if self.particles is None or self.weights is None:
            return

        count = len(self.particles)
        if count <= 0:
            return
        min_particles = max(1, int(self.config.measurement_proposal_min_particles))
        if count < min_particles:
            return
        ratio = float(np.clip(self.config.measurement_proposal_ratio, 0.0, 1.0))
        proposal_count = int(round(ratio * count))
        if ratio > 0.0:
            proposal_count = max(proposal_count, min_particles)
        proposal_count = int(np.clip(proposal_count, 0, max(1, int(0.60 * count))))
        if proposal_count <= 0:
            return

        center = np.asarray(measurement.center_xyz, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(center)):
            return

        indices = self._proposal_replacement_indices(proposal_count)
        offsets = self._measurement_proposal_offsets(proposal_count, measurement, camera_intrinsics)
        self.particles[indices] = center[None, :] + offsets

        self.weights[indices] = max(1.0 / count, float(np.median(self.weights)))
        total = float(np.sum(self.weights))
        if np.isfinite(total) and total > 1e-12:
            self.weights /= total
        else:
            self.weights.fill(1.0 / count)
        self.last_measurement_proposal_used = True

    def _proposal_replacement_indices(self, proposal_count):
        count = len(self.particles)
        weights = np.asarray(self.weights, dtype=np.float64)
        if (
            len(weights) != count
            or not np.all(np.isfinite(weights))
            or float(np.max(weights) - np.min(weights)) <= 1e-12
        ):
            return self.rng.choice(count, size=proposal_count, replace=False)
        return np.argpartition(weights, proposal_count - 1)[:proposal_count]

    def _measurement_proposal_offsets(self, count, measurement, camera_intrinsics=None):
        lateral_std = max(float(self.config.proposal_lateral_std_m), 0.0)
        depth_std = max(float(self.config.proposal_depth_std_m), 0.0)
        ray_basis = self._measurement_ray_basis(measurement, camera_intrinsics)
        if ray_basis is None:
            offsets = self.rng.normal(0.0, lateral_std, (count, 3))
            offsets[:, 2] = self.rng.normal(0.0, depth_std, count)
            return offsets

        ray, basis_x, basis_y = ray_basis
        lateral_x = self.rng.normal(0.0, lateral_std, count)
        lateral_y = self.rng.normal(0.0, lateral_std, count)
        depth = self.rng.normal(0.0, depth_std, count)
        return (
            lateral_x[:, None] * basis_x[None, :]
            + lateral_y[:, None] * basis_y[None, :]
            + depth[:, None] * ray[None, :]
        )

    @staticmethod
    def _measurement_ray_basis(measurement, camera_intrinsics):
        if camera_intrinsics is None or getattr(measurement, "image_center_xy", None) is None:
            return None
        try:
            fx = float(camera_intrinsics["fx"])
            fy = float(camera_intrinsics["fy"])
            cx = float(camera_intrinsics["cx"])
            cy = float(camera_intrinsics["cy"])
            u, v = measurement.image_center_xy
            ray = np.array([(float(u) - cx) / fx, -(float(v) - cy) / fy, 1.0], dtype=np.float64)
        except Exception:
            return None
        ray_norm = np.linalg.norm(ray)
        if not np.isfinite(ray_norm) or ray_norm < 1e-8:
            return None
        ray = ray / ray_norm
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(seed, ray))) > 0.92:
            seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        basis_x = np.cross(seed, ray)
        basis_x_norm = np.linalg.norm(basis_x)
        if basis_x_norm < 1e-8:
            return None
        basis_x = basis_x / basis_x_norm
        basis_y = np.cross(ray, basis_x)
        basis_y = basis_y / max(np.linalg.norm(basis_y), 1e-8)
        return ray, basis_x, basis_y

    def _weight(self, measurement, camera_intrinsics=None):
        center = np.asarray(measurement.center_xyz, dtype=np.float64).reshape(3)

        residual = getattr(measurement, "residual_m", 0.0)
        if not np.isfinite(residual):
            residual = 0.0
        pos_sigma = max(float(self.config.measurement_position_std_m), 2.0 * float(residual), 1e-4)

        center_error_sq = np.sum((self.particles[:, :3] - center) ** 2, axis=1)
        log_likelihood = -0.5 * center_error_sq / (pos_sigma * pos_sigma)

        surface_log_likelihood = self._surface_log_likelihood(measurement)
        if surface_log_likelihood is not None:
            log_likelihood += surface_log_likelihood
            self.last_model_likelihood_used = True

        projection_log_likelihood = self._projection_log_likelihood(measurement, camera_intrinsics)
        if projection_log_likelihood is not None:
            log_likelihood += projection_log_likelihood
            self.last_projection_likelihood_used = True

        log_likelihood -= float(np.max(log_likelihood))
        likelihood = np.exp(log_likelihood)
        weighted = self.weights * likelihood
        total = float(np.sum(weighted))
        if not np.isfinite(total) or total <= 1e-12:
            self._initialize(measurement)
            return

        self.weights = weighted / total

    def _maybe_resample(self, force=False, noise_scale=1.0):
        if self.weights is None or self.particles is None:
            return

        effective = self._effective_sample_size()
        threshold = float(self.config.resample_effective_ratio) * len(self.weights)
        if not force and effective >= threshold:
            return

        indices = self._systematic_resample()
        self.particles = self.particles[indices].copy()
        self.weights.fill(1.0 / len(self.weights))

        noise_scale = float(np.clip(noise_scale, 0.25, self.config.max_noise_scale))
        self.particles += self.rng.normal(
            0.0,
            self.config.process_position_std_m * 0.7 * noise_scale,
            (len(self.particles), 3),
        )

    def _estimate(self, measurement_used, prediction_only):
        if self.particles is None or self.weights is None:
            return None

        center = np.average(self.particles[:, :3], axis=0, weights=self.weights)
        radius = self._current_radius()

        return ParticleFilterResult(
            center_xyz=center.astype(np.float32),
            velocity_xyz=np.zeros(3, dtype=np.float32),
            radius_m=float(radius),
            effective_sample_size=self._effective_sample_size(),
            measurement_used=bool(measurement_used),
            prediction_only=bool(prediction_only),
            lost_frames=int(self.lost_frames),
            model_likelihood_used=bool(self.last_model_likelihood_used),
            projection_likelihood_used=bool(self.last_projection_likelihood_used),
            measurement_proposal_used=bool(self.last_measurement_proposal_used),
            radius_calibrated=bool(self.radius_calibrated),
            radius_sample_count=len(self.radius_samples),
            motion_noise_scale=float(self.last_motion_noise_scale),
        )

    def _should_reinitialize(self, measurement):
        if self.particles is None or self.weights is None:
            return True

        center = np.asarray(measurement.center_xyz, dtype=np.float64).reshape(3)
        current = np.average(self.particles[:, :3], axis=0, weights=self.weights)
        distance = float(np.linalg.norm(center - current))
        threshold = max(float(self.config.reinitialize_distance_m), 3.0 * float(self.config.measurement_position_std_m))
        return distance > threshold

    def _blend_position_measurement(self, measurement):
        if measurement is None or self.particles is None:
            return

        blend = float(np.clip(self.config.position_measurement_blend, 0.0, 1.0))
        if self._has_surface_model_measurement(measurement):
            blend *= float(np.clip(self.config.surface_position_blend_scale, 0.0, 1.0))
        if blend <= 0.0:
            return

        center = np.asarray(measurement.center_xyz, dtype=np.float64).reshape(3)
        self.particles[:, :3] = (1.0 - blend) * self.particles[:, :3] + blend * center

    def _surface_log_likelihood(self, measurement):
        if not bool(self.config.surface_likelihood):
            return None
        if self.particles is None:
            return None

        points, normals, patch_weights = self._surface_likelihood_observation(measurement)
        if len(points) < int(self.config.min_surface_likelihood_points):
            return None

        radius = self._current_radius(default=getattr(measurement, "radius_m", 0.0))
        if radius <= 0.0:
            return None

        centers = self.particles[:, :3]
        distances = np.linalg.norm(points[None, :, :] - centers[:, None, :], axis=2)
        residuals = np.abs(distances - radius)

        point_weights = patch_weights
        patch_weights = np.broadcast_to(point_weights[None, :], residuals.shape).copy()
        trim_fraction = float(np.clip(self.config.surface_trim_fraction, 0.10, 1.0))
        keep_count = int(np.ceil(trim_fraction * residuals.shape[1]))
        keep_count = int(np.clip(keep_count, 1, residuals.shape[1]))
        if keep_count < residuals.shape[1]:
            indices = np.argpartition(residuals, keep_count - 1, axis=1)[:, :keep_count]
            residuals = np.take_along_axis(residuals, indices, axis=1)
            patch_weights = np.take_along_axis(patch_weights, indices, axis=1)

        surface_error = weighted_row_mean(residuals, patch_weights)
        residual = getattr(measurement, "residual_m", 0.0)
        if not np.isfinite(residual):
            residual = 0.0
        sigma = max(float(self.config.surface_distance_std_m), 2.0 * float(residual), 1e-4)
        pseudo_count = max(float(self.config.surface_pseudo_count), 1.0)
        weight = max(float(self.config.surface_likelihood_weight), 0.0)
        if weight <= 0.0:
            return None

        log_likelihood = -0.5 * weight * pseudo_count * (surface_error / sigma) ** 2
        normal_log_likelihood = self._surface_normal_log_likelihood(points, normals, patch_weights=point_weights)
        if normal_log_likelihood is not None:
            log_likelihood += normal_log_likelihood
        return log_likelihood

    def _surface_normal_log_likelihood(self, points, normals, patch_weights=None):
        if not bool(self.config.surface_normal_likelihood):
            return None
        if self.particles is None:
            return None

        normals = np.asarray(normals, dtype=np.float64)
        if normals.ndim != 2 or normals.shape[1] < 3 or len(normals) != len(points):
            return None
        normal_lengths = np.linalg.norm(normals[:, :3], axis=1)
        valid_normals = np.isfinite(normal_lengths) & (normal_lengths > 0.5)
        if np.count_nonzero(valid_normals) < int(self.config.min_surface_likelihood_points):
            return None

        points = points[valid_normals]
        normals = normals[valid_normals, :3] / normal_lengths[valid_normals, None]
        if patch_weights is None:
            weights = np.ones(len(points), dtype=np.float64)
        else:
            weights = np.asarray(patch_weights, dtype=np.float64).reshape(-1)
            if len(weights) != len(valid_normals):
                weights = np.ones(len(valid_normals), dtype=np.float64)
            weights = weights[valid_normals]
            weights[~np.isfinite(weights) | (weights <= 0.0)] = 1.0

        vectors = points[None, :, :] - self.particles[:, None, :3]
        distances = np.linalg.norm(vectors, axis=2)
        valid_distances = np.isfinite(distances) & (distances > 1e-8)
        radial_normals = np.zeros_like(vectors)
        radial_normals[valid_distances] = vectors[valid_distances] / distances[valid_distances, None]

        alignment = np.sum(radial_normals * normals[None, :, :], axis=2)
        alignment = np.clip(np.abs(alignment), 0.0, 1.0)
        normal_error = 1.0 - alignment
        normal_error[~valid_distances] = 1.0

        trim_fraction = float(np.clip(self.config.surface_trim_fraction, 0.10, 1.0))
        keep_count = int(np.ceil(trim_fraction * normal_error.shape[1]))
        keep_count = int(np.clip(keep_count, 1, normal_error.shape[1]))
        weights = np.broadcast_to(weights[None, :], normal_error.shape).copy()
        if keep_count < normal_error.shape[1]:
            indices = np.argpartition(normal_error, keep_count - 1, axis=1)[:, :keep_count]
            normal_error = np.take_along_axis(normal_error, indices, axis=1)
            weights = np.take_along_axis(weights, indices, axis=1)

        mean_normal_error = weighted_row_mean(normal_error, weights)
        sigma = max(float(self.config.surface_normal_std), 1e-3)
        weight = max(float(self.config.surface_normal_weight), 0.0)
        if weight <= 0.0:
            return None
        pseudo_count = max(float(self.config.surface_pseudo_count), 1.0)
        return -0.5 * weight * pseudo_count * (mean_normal_error / sigma) ** 2

    def _projection_log_likelihood(self, measurement, camera_intrinsics):
        if not bool(self.config.projection_likelihood):
            return None
        if self.particles is None or camera_intrinsics is None:
            return None
        if not self._has_projection_measurement(measurement, camera_intrinsics):
            return None

        try:
            fx = float(camera_intrinsics["fx"])
            fy = float(camera_intrinsics["fy"])
            cx = float(camera_intrinsics["cx"])
            cy = float(camera_intrinsics["cy"])
        except Exception:
            return None
        if not all(np.isfinite(value) and abs(value) > 1e-6 for value in (fx, fy)):
            return None

        radius = self._current_radius(default=getattr(measurement, "radius_m", 0.0))
        if radius <= 0.0:
            return None

        image_center = np.asarray(measurement.image_center_xy, dtype=np.float64).reshape(2)
        image_radius = float(measurement.image_radius_px)
        centers = self.particles[:, :3]
        z = centers[:, 2]
        valid = np.isfinite(z) & (z > 1e-4)

        predicted_u = np.full(len(centers), np.inf, dtype=np.float64)
        predicted_v = np.full(len(centers), np.inf, dtype=np.float64)
        predicted_r = np.full(len(centers), np.inf, dtype=np.float64)
        predicted_u[valid] = cx + fx * centers[valid, 0] / z[valid]
        predicted_v[valid] = cy - fy * centers[valid, 1] / z[valid]
        predicted_r[valid] = 0.5 * (abs(fx) + abs(fy)) * radius / z[valid]

        center_error_sq = (predicted_u - image_center[0]) ** 2 + (predicted_v - image_center[1]) ** 2
        radius_error_sq = (predicted_r - image_radius) ** 2
        center_sigma = max(float(self.config.projection_center_std_px), 1e-3)
        radius_sigma = max(float(self.config.projection_radius_std_px), 1e-3)
        weight = max(float(self.config.projection_likelihood_weight), 0.0)
        if weight <= 0.0:
            return None

        log_likelihood = -0.5 * weight * center_error_sq / (center_sigma * center_sigma)
        log_likelihood += -0.5 * weight * radius_error_sq / (radius_sigma * radius_sigma)
        log_likelihood[~valid] = -1e6
        return log_likelihood

    def _surface_likelihood_observation(self, measurement):
        if measurement is None:
            empty_points = np.empty((0, 3), dtype=np.float64)
            return empty_points, empty_points, np.empty((0,), dtype=np.float64)

        raw_points = getattr(measurement, "surface_points", None)
        if raw_points is None:
            empty_points = np.empty((0, 3), dtype=np.float64)
            return empty_points, empty_points, np.empty((0,), dtype=np.float64)

        points = np.asarray(raw_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 3:
            empty_points = np.empty((0, 3), dtype=np.float64)
            return empty_points, empty_points, np.empty((0,), dtype=np.float64)

        points = points[:, :3]
        normals = getattr(measurement, "surface_normals", None)
        if normals is None:
            normals = np.zeros_like(points, dtype=np.float64)
        else:
            normals = np.asarray(normals, dtype=np.float64)
            if normals.ndim != 2 or normals.shape[1] < 3 or len(normals) != len(points):
                normals = np.zeros_like(points, dtype=np.float64)
            else:
                normals = normals[:, :3]

        weights = getattr(measurement, "surface_weights", None)
        if weights is None:
            weights = np.ones(len(points), dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64).reshape(-1)
            if len(weights) != len(points):
                weights = np.ones(len(points), dtype=np.float64)

        valid = np.all(np.isfinite(points), axis=1)
        valid &= np.all(np.isfinite(normals), axis=1)
        valid &= np.isfinite(weights)
        points = points[valid]
        normals = normals[valid]
        weights = weights[valid]

        normal_lengths = np.linalg.norm(normals, axis=1)
        valid_normals = normal_lengths > 1e-6
        normals[valid_normals] /= normal_lengths[valid_normals, None]
        normals[~valid_normals] = 0.0

        if bool(self.config.surface_area_weighting):
            positive = weights[np.isfinite(weights) & (weights > 0.0)]
            if len(positive) >= 8:
                low, high = np.percentile(positive, [5.0, 95.0])
                if np.isfinite(low) and np.isfinite(high) and high > low:
                    weights = np.clip(weights, low, high)
            fallback = float(np.median(positive)) if len(positive) else 1.0
            weights[~np.isfinite(weights) | (weights <= 0.0)] = fallback
        else:
            weights = np.ones(len(points), dtype=np.float64)

        max_points = max(int(self.config.surface_likelihood_points), 0)
        if max_points > 0 and len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
            points = points[indices]
            normals = normals[indices]
            weights = weights[indices]

        return (
            np.ascontiguousarray(points, dtype=np.float64),
            np.ascontiguousarray(normals, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.float64),
        )

    def _surface_likelihood_points(self, measurement):
        points, _normals, _weights = self._surface_likelihood_observation(measurement)
        return points

    def _has_surface_model_measurement(self, measurement):
        points = getattr(measurement, "surface_points", None)
        if points is None:
            return False
        try:
            return len(points) >= int(self.config.min_surface_likelihood_points)
        except TypeError:
            return False

    def _has_projection_measurement(self, measurement, camera_intrinsics):
        if measurement is None or camera_intrinsics is None:
            return False
        if getattr(measurement, "image_center_xy", None) is None:
            return False
        image_radius = getattr(measurement, "image_radius_px", None)
        try:
            return np.isfinite(float(image_radius)) and float(image_radius) > 0.0
        except Exception:
            return False

    def _adaptive_noise_scale(self, measurement):
        if not bool(self.config.adaptive_noise):
            return 1.0

        scale = 1.0
        residual = getattr(measurement, "residual_m", 0.0)
        if np.isfinite(residual):
            scale += float(self.config.residual_noise_gain) * max(float(residual), 0.0)

        return float(np.clip(scale, 0.5, self.config.max_noise_scale))

    def _lost_noise_scale(self):
        if not bool(self.config.adaptive_noise):
            return 1.0
        scale = 1.0 + float(self.config.lost_noise_gain) * max(self.lost_frames, 0)
        return float(np.clip(scale, 1.0, self.config.max_noise_scale))

    def _update_radius_from_measurement(self, measurement):
        fixed_radius = self._configured_fixed_radius()
        if self.config.lock_radius and fixed_radius is not None:
            self.radius_m = fixed_radius
            self.radius_calibrated = True
            return

        measured = self._valid_radius(getattr(measurement, "radius_m", 0.0))
        if measured is None:
            return

        if not self.config.lock_radius:
            self.radius_m = measured
            self.radius_calibrated = False
            return

        if not self.radius_calibrated:
            self.radius_samples.append(measured)
            max_samples = max(int(self.config.radius_calibration_frames) * 2, int(self.config.radius_calibration_min_frames))
            if len(self.radius_samples) > max_samples:
                self.radius_samples = self.radius_samples[-max_samples:]
            self.radius_m = self._robust_radius(self.radius_samples, fallback=measured)
            self._maybe_lock_calibrated_radius()

    def _maybe_lock_calibrated_radius(self):
        sample_count = len(self.radius_samples)
        min_count = max(1, int(self.config.radius_calibration_min_frames))
        target_count = max(min_count, int(self.config.radius_calibration_frames))
        if sample_count < min_count:
            return

        samples = np.asarray(self.radius_samples, dtype=np.float64)
        center = float(np.median(samples))
        mad = float(np.median(np.abs(samples - center)))
        tolerance = max(float(self.config.radius_calibration_relative_mad) * max(center, 1e-6), 3.0 * self.config.measurement_radius_std_m)
        inliers = samples[np.abs(samples - center) <= tolerance]
        if len(inliers) < min_count:
            return

        robust_radius = self._robust_radius(inliers, fallback=center)
        stable = mad <= tolerance
        enough = sample_count >= target_count
        if stable or enough:
            self.radius_m = robust_radius
            self.radius_calibrated = True
            self.radius_samples = [float(value) for value in inliers]

    def _robust_radius(self, samples, fallback=0.0):
        samples = np.asarray(samples, dtype=np.float64)
        samples = samples[np.isfinite(samples)]
        samples = samples[(samples >= self.config.min_radius_m) & (samples <= self.config.max_radius_m)]
        if len(samples) == 0:
            valid = self._valid_radius(fallback)
            return 0.0 if valid is None else valid
        return float(np.median(samples))

    def _current_radius(self, default=0.0):
        valid = self._valid_radius(self.radius_m)
        if valid is not None:
            return valid
        valid = self._valid_radius(default)
        if valid is not None:
            return valid
        return 0.0

    def _configured_fixed_radius(self):
        return self._valid_radius(getattr(self.config, "fixed_radius_m", 0.0))

    def _valid_radius(self, radius):
        try:
            radius = float(radius)
        except Exception:
            return None
        if np.isfinite(radius) and self.config.min_radius_m <= radius <= self.config.max_radius_m:
            return float(radius)
        return None

    def _systematic_resample(self):
        count = len(self.weights)
        positions = (self.rng.random() + np.arange(count)) / count
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        return np.searchsorted(cumulative, positions, side="left")

    def _effective_sample_size(self):
        return float(1.0 / max(np.sum(self.weights * self.weights), 1e-12))

    @staticmethod
    def _valid_measurement(measurement):
        center = np.asarray(measurement.center_xyz, dtype=np.float64).reshape(-1)
        radius = float(measurement.radius_m)
        return len(center) >= 3 and np.all(np.isfinite(center[:3])) and np.isfinite(radius) and radius > 0.0


def weighted_row_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape:
        return np.mean(values, axis=1)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    safe_values = np.where(valid, values, 0.0)
    safe_weights = np.where(valid, weights, 0.0)
    totals = np.sum(safe_weights, axis=1)
    weighted = np.sum(safe_values * safe_weights, axis=1)
    fallback = np.mean(values, axis=1)
    return np.where(totals > 1e-12, weighted / np.maximum(totals, 1e-12), fallback)


def filtered_sphere_estimate(measurement, result):
    if result is None:
        return None

    if measurement is not None:
        surface_points = measurement.surface_points
        residual = measurement.residual_m
        table_normal = measurement.table_normal
        table_offset = measurement.table_offset
        base_method = measurement.method
        image_center_xy = getattr(measurement, "image_center_xy", None)
        image_radius_px = getattr(measurement, "image_radius_px", None)
    else:
        surface_points = np.empty((0, 3), dtype=np.float32)
        residual = 0.0
        table_normal = None
        table_offset = None
        base_method = "no measurement"
        image_center_xy = None
        image_radius_px = None

    mode = "prediction" if result.prediction_only else "update"
    likelihoods = []
    if result.model_likelihood_used:
        likelihoods.append("surface")
    if result.projection_likelihood_used:
        likelihoods.append("rgb")
    likelihood_mode = "+".join(likelihoods) if likelihoods else "center"
    radius_mode = "locked" if result.radius_calibrated else f"calib {result.radius_sample_count}"
    method = (
        f"particle filter ball {mode} {likelihood_mode} | {radius_mode} r | "
        f"{base_method} | ess={result.effective_sample_size:.0f} "
        f"lost={result.lost_frames} noise={result.motion_noise_scale:.1f}"
    )
    return SphereEstimate3D(
        center_xyz=np.asarray(result.center_xyz, dtype=np.float32),
        radius_m=float(result.radius_m),
        surface_points=np.asarray(surface_points, dtype=np.float32),
        residual_m=float(residual),
        method=method,
        table_normal=table_normal,
        table_offset=table_offset,
        image_center_xy=image_center_xy,
        image_radius_px=image_radius_px,
    )
