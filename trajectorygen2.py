import numpy as np
from typing import Callable, Optional, Dict, Any

class KSpaceTrajectoryGenerator:
    def __init__(
        self,
        fov=0.24,
        resolution=0.001,
        dt=4e-6,
        g_max=40e-3,
        s_max=150.0,
        n_interleaves=8,
        gamma=42.576e6,
        traj_type='spiral',
        turns=1,
        ramp_fraction=0.1,
        add_rewinder=True,
        add_spoiler=False,
        add_slew_limited_ramps=True,
        dim=2,
        n_stacks: Optional[int] = None,
        zmax: Optional[float] = None,
        custom_traj_func: Optional[Callable[..., Any]] = None,
        per_interleaf_params: Optional[Dict[int, Dict[str, Any]]] = None,
        time_varying_params: Optional[Callable[[float], Dict[str, float]]] = None,
    ):
        """
        Initialize trajectory generator with imaging and hardware parameters.

        Additional parameters for 3D and custom shapes:
        - dim: 2 or 3 (for 2D or 3D)
        - n_stacks: number of stacks (z-slices) for 3D stack-of-spirals/stars
        - zmax: maximum kz value for 3D
        - custom_traj_func: function handle for user-defined custom trajectory
        - per_interleaf_params: dict of per-interleaf parameter overrides
        - time_varying_params: function of t returning dict of param values
        """
        self.fov = fov
        self.resolution = resolution
        self.dt = dt
        self.g_max = g_max
        self.s_max = s_max
        self.n_interleaves = n_interleaves
        self.gamma = gamma
        self.traj_type = traj_type
        self.turns = turns
        self.ramp_fraction = ramp_fraction
        self.add_rewinder = add_rewinder
        self.add_spoiler = add_spoiler
        self.add_slew_limited_ramps = add_slew_limited_ramps
        self.dim = dim
        self.n_stacks = n_stacks
        self.zmax = zmax
        self.custom_traj_func = custom_traj_func
        self.per_interleaf_params = per_interleaf_params or {}
        self.time_varying_params = time_varying_params

        # Derived quantities
        self.k_max = 1 / (2 * self.resolution)
        self.g_required = min(self.k_max / (self.gamma * self.dt), self.g_max)
        self.n_samples = int(np.ceil((self.k_max * 2 * np.pi * self.fov) / (self.gamma * self.g_required * self.dt)))
        self.n_samples = max(self.n_samples, 1)
        self.ramp_samples = int(np.ceil(self.ramp_fraction * self.n_samples))
        self.flat_samples = self.n_samples - 2 * self.ramp_samples

    def _slew_limited_ramp(self, N, sign=1):
        t_ramp = np.linspace(0, 1, N)
        ramp = 0.5 * (1 - np.cos(np.pi * t_ramp))  # [0,1], smooth
        return sign * ramp

    def _make_radius_profile(self, n_samples=None):
        n_samples = n_samples or self.n_samples
        ramp_samples = int(self.ramp_fraction * n_samples)
        flat_samples = n_samples - 2 * ramp_samples
        if self.add_slew_limited_ramps:
            ramp_up = self._slew_limited_ramp(ramp_samples)
            flat = np.ones(flat_samples)
            ramp_down = 1 - self._slew_limited_ramp(ramp_samples)
            r_profile = np.concatenate([ramp_up, flat, ramp_down])
        else:
            r_profile = np.ones(n_samples)
            r_profile[:ramp_samples] = np.linspace(0, 1, ramp_samples)
            r_profile[-ramp_samples:] = np.linspace(1, 0, ramp_samples)
        return r_profile

    def _enforce_gradient_limits(self, gx, gy, gz=None):
        g_norm = np.sqrt(gx ** 2 + gy ** 2 + (gz**2 if gz is not None else 0))
        over_gmax = g_norm > self.g_max
        if np.any(over_gmax):
            scale = self.g_max / np.max(g_norm)
            gx[over_gmax] *= scale
            gy[over_gmax] *= scale
            if gz is not None:
                gz[over_gmax] *= scale

        slew = np.sqrt(np.gradient(gx, self.dt) ** 2 +
                       np.gradient(gy, self.dt) ** 2 +
                       (np.gradient(gz, self.dt) ** 2 if gz is not None else 0))
        over_smax = slew > self.s_max
        if np.any(over_smax):
            scale = self.s_max / np.max(slew)
            gx[over_smax] *= scale
            gy[over_smax] *= scale
            if gz is not None:
                gz[over_smax] *= scale
        return gx, gy, gz

    # 2D/3D Standard Trajectories
    def _generate_standard(self, interleaf_idx, t, n_samples, **params):
        # Per-interleaf overrides
        local_params = {**self.__dict__, **params}
        fov = local_params.get("fov", self.fov)
        resolution = local_params.get("resolution", self.resolution)
        turns = local_params.get("turns", self.turns)
        k_max = 1 / (2 * resolution)
        r_profile = self._make_radius_profile(n_samples)
        # Time-varying parameter support
        if self.time_varying_params is not None:
            # param_func = lambda ti: self.time_varying_params(ti)
            for i, ti in enumerate(t):
                for key, val in self.time_varying_params(ti).items():
                    if key == "fov":
                        fov = val
                    if key == "resolution":
                        resolution = val
                k_max = 1 / (2 * resolution)
                r_profile[i] = min(r_profile[i], k_max / self.k_max)

        # 2D
        if self.dim == 2:
            if self.traj_type == "spiral":
                phi = 2 * np.pi * interleaf_idx / self.n_interleaves
                theta = turns * 2 * np.pi * t / t[-1] + phi
                r = r_profile * k_max
                kx = r * np.cos(theta)
                ky = r * np.sin(theta)
                kz = None
            elif self.traj_type == "radial":
                angle = np.pi * interleaf_idx / self.n_interleaves
                k_line = np.linspace(-k_max, k_max, n_samples) * r_profile
                kx = k_line * np.cos(angle)
                ky = k_line * np.sin(angle)
                kz = None
            elif self.traj_type == "epi":
                # Example: simple EPI (rectilinear, zigzag)
                kx = np.linspace(-k_max, k_max, n_samples)
                ky = np.zeros(n_samples)
                kz = None
            elif self.traj_type == "rosette":
                f1 = params.get("f1", 5)
                f2 = params.get("f2", 7)
                a = params.get("a", 0.5)
                phase = 2*np.pi*interleaf_idx/self.n_interleaves
                tt = np.linspace(0, 2*np.pi, n_samples)
                kx = k_max * (a * np.sin(f1*tt+phase) + (1-a) * np.sin(f2*tt+phase))
                ky = k_max * (a * np.cos(f1*tt+phase) + (1-a) * np.cos(f2*tt+phase))
                kz = None
            else:
                raise ValueError(f"Unknown 2D traj_type {self.traj_type}")
            gx = np.gradient(kx, self.dt) / self.gamma
            gy = np.gradient(ky, self.dt) / self.gamma
            gz = None
        # 3D
        elif self.dim == 3:
            # Stack of Spirals
            if self.traj_type == "stackofspirals":
                n_stacks = self.n_stacks or 8
                zmax = self.zmax or k_max
                stack_idx = interleaf_idx // self.n_interleaves
                slice_idx = interleaf_idx % self.n_interleaves
                z_locations = np.linspace(-zmax, zmax, n_stacks)
                z = z_locations[stack_idx]
                phi = 2 * np.pi * slice_idx / self.n_interleaves
                theta = turns * 2 * np.pi * t / t[-1] + phi
                r = r_profile * k_max
                kx = r * np.cos(theta)
                ky = r * np.sin(theta)
                kz = np.ones(n_samples) * z
            # Phyllotaxis 3D
            elif self.traj_type == "phyllotaxis":
                golden_angle = np.pi * (3 - np.sqrt(5))
                theta = golden_angle * interleaf_idx
                z = np.linspace(1 - 1/n_samples, -1 + 1/n_samples, n_samples)
                radius = np.sqrt(1 - z**2)
                kx = k_max * radius * np.cos(theta)
                ky = k_max * radius * np.sin(theta)
                kz = k_max * z
            # Cones
            elif self.traj_type == "cones":
                phi = 2 * np.pi * interleaf_idx / self.n_interleaves
                tt = np.linspace(0, 1, n_samples)
                theta = np.arccos(1 - 2*tt)
                kx = k_max * tt * np.sin(theta) * np.cos(phi)
                ky = k_max * tt * np.sin(theta) * np.sin(phi)
                kz = k_max * tt * np.cos(theta)
            # 3D radial
            elif self.traj_type == "radial3d":
                phi = 2 * np.pi * interleaf_idx / self.n_interleaves
                theta = np.arccos(1 - 2*interleaf_idx/self.n_interleaves)
                k_line = np.linspace(-k_max, k_max, n_samples)
                kx = k_line * np.sin(theta) * np.cos(phi)
                ky = k_line * np.sin(theta) * np.sin(phi)
                kz = k_line * np.cos(theta)
            else:
                raise ValueError(f"Unknown 3D traj_type {self.traj_type}")
            gx = np.gradient(kx, self.dt) / self.gamma
            gy = np.gradient(ky, self.dt) / self.gamma
            gz = np.gradient(kz, self.dt) / self.gamma
        else:
            raise ValueError("dim must be 2 or 3")
        gx, gy, gz = self._enforce_gradient_limits(gx, gy, gz)
        return kx, ky, kz, gx, gy, gz

    def generate(self):
        n_interleaves = self.n_interleaves
        n_samples = self.n_samples
        t = np.arange(n_samples) * self.dt

        # Determine dimensionality
        if self.dim == 2:
            kx = np.zeros((n_interleaves, n_samples))
            ky = np.zeros((n_interleaves, n_samples))
            gx = np.zeros((n_interleaves, n_samples))
            gy = np.zeros((n_interleaves, n_samples))
            kz = gz = None
        else:
            kx = np.zeros((n_interleaves, n_samples))
            ky = np.zeros((n_interleaves, n_samples))
            kz = np.zeros((n_interleaves, n_samples))
            gx = np.zeros((n_interleaves, n_samples))
            gy = np.zeros((n_interleaves, n_samples))
            gz = np.zeros((n_interleaves, n_samples))

        for idx in range(n_interleaves):
            # Per-interleaf param overrides
            params = self.per_interleaf_params.get(idx, {})
            # Custom trajectory plugin
            if self.custom_traj_func is not None:
                k_vals, g_vals = self.custom_traj_func(idx, t, n_samples, **params)
                kx[idx], ky[idx] = k_vals[:2]
                gx[idx], gy[idx] = g_vals[:2]
                if self.dim == 3 and len(k_vals) > 2:
                    kz[idx] = k_vals[2]
                    gz[idx] = g_vals[2]
                continue
            # Standard trajectory
            kx_i, ky_i, kz_i, gx_i, gy_i, gz_i = self._generate_standard(idx, t, n_samples, **params)
            kx[idx], ky[idx] = kx_i, ky_i
            gx[idx], gy[idx] = gx_i, gy_i
            if self.dim == 3:
                kz[idx] = kz_i
                gz[idx] = gz_i

        # TODO: Add spoiler and rewinder for 3D, if needed

        t = np.arange(kx.shape[1]) * self.dt
        if self.dim == 2:
            return kx, ky, gx, gy, t
        else:
            return kx, ky, kz, gx, gy, gz, t

    # Example interface for user plugin
    @staticmethod
    def plugin_example(idx, t, n_samples, **kwargs):
        # Example: circle trajectory
        kx = np.cos(2 * np.pi * t / t[-1])
        ky = np.sin(2 * np.pi * t / t[-1])
        gx = np.gradient(kx, t)
        gy = np.gradient(ky, t)
        return (kx, ky), (gx, gy)

    def check_gradient_and_slew_limits(self, k_traj):
        gamma = self.gamma
        G = np.diff(k_traj, axis=0) / self.dt / gamma
        slew = np.diff(G, axis=0) / self.dt
        grad_ok = np.all(np.abs(G) <= self.g_max)
        slew_ok = np.all(np.abs(slew) <= self.s_max)
        return grad_ok, slew_ok, G, slew

# Example usage:
# gen = KSpaceTrajectoryGenerator(
#     fov=0.24, resolution=0.001, dt=4e-6, g_max=40e-3, s_max=150.0,
#     n_interleaves=16, traj_type='stackofspirals', dim=3, n_stacks=8, zmax=1.5,
#     ramp_fraction=0.1, add_rewinder=True, add_spoiler=False, add_slew_limited_ramps=True
# )
# kx, ky, kz, gx, gy, gz, t = gen.generate()
#
# # Or with a custom plugin:
# gen2 = KSpaceTrajectoryGenerator(
#     n_interleaves=1, traj_type='custom', custom_traj_func=KSpaceTrajectoryGenerator.plugin_example
# )
# kx, ky, gx, gy, t = gen2.generate()
