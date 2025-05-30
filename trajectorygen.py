import numpy as np



def radial_trajectory(num_spokes, num_points, fov, dwell_time):
    angles = np.linspace(0, np.pi, num_spokes, endpoint=False)
    k = np.zeros((num_spokes, num_points, 2))
    max_k = 1/(2*fov)
    for i, theta in enumerate(angles):
        kx = np.linspace(-max_k, max_k, num_points) * np.cos(theta)
        ky = np.linspace(-max_k, max_k, num_points) * np.sin(theta)
        k[i, :, 0] = kx
        k[i, :, 1] = ky
    return k

def spiral_trajectory(num_arms, num_points, fov, dwell_time, turns=1):
    k = np.zeros((num_arms, num_points, 2))
    max_k = 1/(2*fov)
    for i in range(num_arms):
        phi = 2 * np.pi * i / num_arms
        t = np.linspace(0, 1, num_points)
        r = t * max_k
        theta = turns * 2 * np.pi * t + phi
        kx = r * np.cos(theta)
        ky = r * np.sin(theta)
        k[i, :, 0] = kx
        k[i, :, 1] = ky
    return k

import numpy as np

def generate_spiral_trajectory(
    fov=0.24,               # Field of view in meters
    resolution=0.001,       # Desired resolution in meters
    dt=4e-6,                # Sample time in seconds
    g_max=40e-3,            # Maximum gradient amplitude in T/m
    s_max=150.0,            # Maximum slew rate in T/m/s
    n_interleaves=8,         # Number of interleaves
    gamma = 42.576e6  # Gyromagnetic ratio in Hz/T
):
    """
    Generate a spiral k-space trajectory based on imaging specifications.

    Parameters:
    - fov: Field of view (m)
    - resolution: Desired resolution (m)
    - dt: Sample time (s)
    - g_max: Maximum gradient amplitude (T/m)
    - s_max: Maximum slew rate (T/m/s)
    - n_interleaves: Number of interleaves

    Returns:
    - kx, ky: k-space trajectory components
    - gx, gy: Gradient waveform components
    - t: Time vector
    """


    # Calculate maximum k-space radius
    k_max = 1 / (2 * resolution)

    # Time to reach k_max at maximum gradient amplitude
    g_required = k_max / (gamma * dt)
    if g_required > g_max:
        g_required = g_max
        print("Warning: Desired resolution exceeds maximum gradient amplitude. Adjusting gradient amplitude.")

    # Time to reach g_required at maximum slew rate
    t_ramp = g_required / s_max

    # Total number of samples
    n_samples = int(np.ceil((k_max * 2 * np.pi * fov) / (gamma * g_required * dt)))

    # Time vector
    t = np.arange(n_samples) * dt

    # Angular position
    theta = 2 * np.pi * n_interleaves * t / t[-1]

    # Radius as a function of time
    r = (g_required * gamma * t) / (2 * np.pi)

    # k-space trajectory
    kx = r * np.cos(theta)
    ky = r * np.sin(theta)

    # Gradient waveforms
    gx = np.gradient(kx, dt) / gamma
    gy = np.gradient(ky, dt) / gamma

    return kx, ky, gx, gy, t


def stack_of_spirals(num_arms, num_points, num_stacks, fov, zmax, turns=1):
    """
    3D Stack of Spirals:
    For each z slice, draws a spiral in the (kx, ky) plane, with kz constant for the stack.
    Returns [num_stacks, num_arms, num_points, 3] array.
    """
    k = np.zeros((num_stacks, num_arms, num_points, 3))
    z_locations = np.linspace(-zmax, zmax, num_stacks)
    max_k = 1/(2*fov)
    for iz, z in enumerate(z_locations):
        for i in range(num_arms):
            phi = 2 * np.pi * i / num_arms
            t = np.linspace(0, 1, num_points)
            r = t * max_k
            theta = turns * 2 * np.pi * t + phi
            kx = r * np.cos(theta)
            ky = r * np.sin(theta)
            kz = np.ones(num_points) * z
            k[iz, i, :, 0] = kx
            k[iz, i, :, 1] = ky
            k[iz, i, :, 2] = kz
    return k

def stack_of_stars(num_spokes, num_points, num_stacks, fov, zmax):
    k = np.zeros((num_stacks, num_spokes, num_points, 3))
    z_locations = np.linspace(-zmax, zmax, num_stacks)
    max_k = 1/(2*fov)
    for iz, z in enumerate(z_locations):
        for i, theta in enumerate(np.linspace(0, np.pi, num_spokes, endpoint=False)):
            kx = np.linspace(-max_k, max_k, num_points) * np.cos(theta)
            ky = np.linspace(-max_k, max_k, num_points) * np.sin(theta)
            kz = np.ones(num_points) * z
            k[iz, i, :, 0] = kx
            k[iz, i, :, 1] = ky
            k[iz, i, :, 2] = kz
    return k

def cones_trajectory(num_cones, num_points, fov, zmax):
    k = np.zeros((num_cones, num_points, 3))
    max_k = 1/(2*fov)
    for i in range(num_cones):
        phi = 2 * np.pi * i / num_cones
        t = np.linspace(0, 1, num_points)
        theta = np.arccos(1 - 2*t)
        kx = max_k * t * np.sin(theta) * np.cos(phi)
        ky = max_k * t * np.sin(theta) * np.sin(phi)
        kz = max_k * t * np.cos(theta)
        k[i, :, 0] = kx
        k[i, :, 1] = ky
        k[i, :, 2] = kz
    return k

def cones_3d_trajectory(num_cones, num_points, fov):
    """3D cones trajectory: Each cone defined by its axis on a sphere and samples along the cone."""
    k = []
    max_k = 1/(2*fov)
    # Distribute cones uniformly on sphere surface (using spherical Fibonacci lattice)
    indices = np.arange(0, num_cones, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_cones)
    theta = np.pi * (1 + 5**0.5) * indices
    for i in range(num_cones):
        axis = [
            np.sin(phi[i]) * np.cos(theta[i]),
            np.sin(phi[i]) * np.sin(theta[i]),
            np.cos(phi[i])
        ]
        t = np.linspace(0, 1, num_points)
        angle = 2 * np.pi * t
        for j in range(num_points):
            offset = max_k * t[j]
            direction = np.array(axis)
            # Create orthogonal vectors
            if np.allclose(direction, [0,0,1]):
                ortho = np.array([1,0,0])
            else:
                ortho = np.cross(direction, [0,0,1])
                ortho = ortho / np.linalg.norm(ortho)
            perp = np.cross(direction, ortho)
            point = (offset * direction +
                     np.cos(angle[j]) * 0.05*max_k * ortho +
                     np.sin(angle[j]) * 0.05*max_k * perp)
            k.append(point)
    k = np.array(k).reshape(num_cones, num_points, 3)
    return k

def magic_angle_3d_radial(num_spokes, num_points, fov, dwell_time):
    """3D radial with orientations sampling the magic angle for susceptibility imaging."""
    k = np.zeros((num_spokes, num_points, 3))
    max_k = 1/(2*fov)
    # Magic angle = arccos(1/sqrt(3)) ≈ 54.74°
    magic_theta = np.arccos(1/np.sqrt(3))
    phis = np.linspace(0, 2*np.pi, num_spokes, endpoint=False)
    for i, phi in enumerate(phis):
        kx = np.linspace(-max_k, max_k, num_points) * np.sin(magic_theta) * np.cos(phi)
        ky = np.linspace(-max_k, max_k, num_points) * np.sin(magic_theta) * np.sin(phi)
        kz = np.linspace(-max_k, max_k, num_points) * np.cos(magic_theta)
        k[i, :, 0] = kx
        k[i, :, 1] = ky
        k[i, :, 2] = kz
    return k

def rosette_trajectory(num_petals, num_points, fov, dwell_time, a=0.5, f1=5, f2=7):
    """
    2D rosette: kx = a*sin(f1*t) + (1-a)*sin(f2*t)
                ky = a*cos(f1*t) + (1-a)*cos(f2*t)
    """
    max_k = 1/(2*fov)
    t = np.linspace(0, 2*np.pi, num_points)
    k = np.zeros((num_petals, num_points, 2))
    for i in range(num_petals):
        phase = 2*np.pi*i/num_petals
        kx = max_k * (a * np.sin(f1*t+phase) + (1-a) * np.sin(f2*t+phase))
        ky = max_k * (a * np.cos(f1*t+phase) + (1-a) * np.cos(f2*t+phase))
        k[i, :, 0] = kx
        k[i, :, 1] = ky
    return k

def rosette_3d_trajectory(num_petals, num_points, fov, dwell_time, a=0.5, f1=5, f2=7):
    """
    3D Rosette: Spherical extension of 2D rosette
    """
    max_k = 1/(2*fov)
    k = np.zeros((num_petals, num_points, 3))
    t = np.linspace(0, 2*np.pi, num_points)
    for i in range(num_petals):
        phase = 2*np.pi*i/num_petals
        # Trajectory on a sphere, varying both polar and azimuthal angles
        theta = np.pi/2 + a * np.sin(f1*t+phase)
        phi = phase + (1-a) * np.cos(f2*t+phase)
        r = max_k * np.ones_like(t)
        kx = r * np.sin(theta) * np.cos(phi)
        ky = r * np.sin(theta) * np.sin(phi)
        kz = r * np.cos(theta)
        k[i, :, 0] = kx
        k[i, :, 1] = ky
        k[i, :, 2] = kz
    return k

def phyllotaxis_3d_trajectory(num_points, fov):
    """
    3D phyllotaxis (spherical spiral): Uniformly distributes points on a sphere using the golden angle.
    Useful for 3D radial or cones trajectories.
    Returns a [num_points, 3] array.
    """
    max_k = 1/(2*fov)
    golden_angle = np.pi * (3 - np.sqrt(5))  # ~2.399 rad
    indices = np.arange(num_points)
    theta = golden_angle * indices            # azimuthal angle
    z = np.linspace(1 - 1/num_points, -1 + 1/num_points, num_points)  # height [-1, 1]
    radius = np.sqrt(1 - z**2)
    kx = max_k * radius * np.cos(theta)
    ky = max_k * radius * np.sin(theta)
    kz = max_k * z
    k = np.stack([kx, ky, kz], axis=1)
    return k

def trajectory_with_constraints(
    fov=0.24,               # Field of view in meters
    resolution=0.001,       # Desired resolution in meters
    dt=4e-6,                # Sample time in seconds
    g_max=40e-3,            # Maximum gradient amplitude in T/m
    s_max=150.0,            # Maximum slew rate in T/m/s
    n_interleaves=8,        # Number of interleaves for spiral/radial
    gamma=42.576e6,         # Gyromagnetic ratio in Hz/T
    traj_type='spiral',     # 'spiral' or 'radial'
    turns=1,                # Number of turns (for spiral)
    ramp_fraction=0.1,      # Fraction of samples for ramp up/down
    add_rewinder=True,      # Add rewinder gradients at end
):
    """
    Generate a 2D k-space trajectory (spiral or radial) with hardware and imaging constraints,
    including ramp up/down and optional rewinder gradients.

    Parameters:
    - fov: Field of view (m)
    - resolution: Desired resolution (m)
    - dt: Sample time (s)
    - g_max: Maximum gradient amplitude (T/m)
    - s_max: Maximum slew rate (T/m/s)
    - n_interleaves: Number of interleaves (spiral/radial)
    - gamma: Gyromagnetic ratio (Hz/T)
    - traj_type: 'spiral' or 'radial'
    - turns: Number of turns for spiral
    - ramp_fraction: Fraction of total samples for ramp up/down (default 0.1)
    - add_rewinder: If True, append rewinder gradients at end

    Returns:
    - kx, ky: k-space trajectory components [n_interleaves, n_samples(+rewinder)]
    - gx, gy: Gradient waveform components [n_interleaves, n_samples(+rewinder)]
    - t: Time vector
    """

    # Calculate maximum k-space radius (Nyquist)
    k_max = 1 / (2 * resolution)

    # Estimate total readout duration and samples
    g_required = k_max / (gamma * dt)
    if g_required > g_max:
        g_required = g_max

    n_samples = int(np.ceil((k_max * 2 * np.pi * fov) / (gamma * g_required * dt)))
    if n_samples < 1:
        n_samples = 1

    ramp_samples = int(np.ceil(ramp_fraction * n_samples))
    flat_samples = n_samples - 2 * ramp_samples
    t = np.arange(n_samples) * dt

    kx = np.zeros((n_interleaves, n_samples))
    ky = np.zeros((n_interleaves, n_samples))
    gx = np.zeros((n_interleaves, n_samples))
    gy = np.zeros((n_interleaves, n_samples))

    if traj_type == 'spiral':
        for arm in range(n_interleaves):
            phi = 2 * np.pi * arm / n_interleaves

            # Radius profile: ramp up, flat, ramp down
            r_profile = np.ones(n_samples)
            # Ramp up
            r_profile[:ramp_samples] = np.linspace(0, 1, ramp_samples)
            # Ramp down
            r_profile[-ramp_samples:] = np.linspace(1, 0, ramp_samples)

            theta = turns * 2 * np.pi * t / t[-1] + phi
            r = r_profile * k_max
            kx[arm] = r * np.cos(theta)
            ky[arm] = r * np.sin(theta)
            gx[arm] = np.gradient(kx[arm], dt) / gamma
            gy[arm] = np.gradient(ky[arm], dt) / gamma

    elif traj_type == 'radial':
        for spoke in range(n_interleaves):
            angle = np.pi * spoke / n_interleaves
            # k goes from -k_max to +k_max with ramp up/down
            k_line = np.linspace(-k_max, k_max, n_samples)
            ramp = np.ones(n_samples)
            ramp[:ramp_samples] = np.linspace(0, 1, ramp_samples)
            ramp[-ramp_samples:] = np.linspace(1, 0, ramp_samples)
            k_line = k_line * ramp
            kx[spoke] = k_line * np.cos(angle)
            ky[spoke] = k_line * np.sin(angle)
            gx[spoke] = np.gradient(kx[spoke], dt) / gamma
            gy[spoke] = np.gradient(ky[spoke], dt) / gamma

    else:
        raise ValueError("traj_type must be 'spiral' or 'radial'.")

    # Optionally add rewinder gradients (return to (0,0) in k-space)
    if add_rewinder:
        kx_rw = []
        ky_rw = []
        gx_rw = []
        gy_rw = []
        for arm in range(n_interleaves):
            # Calculate final k-space position
            k_end = np.array([kx[arm, -1], ky[arm, -1]])
            # Linear rewinder in N_rw points
            N_rw = ramp_samples
            k_rewind = np.linspace(k_end, [0, 0], N_rw)
            gx_rewind = np.gradient(k_rewind[:, 0], dt) / gamma
            gy_rewind = np.gradient(k_rewind[:, 1], dt) / gamma

            # Concatenate to original
            kx_rw.append(np.concatenate([kx[arm], k_rewind[:, 0]]))
            ky_rw.append(np.concatenate([ky[arm], k_rewind[:, 1]]))
            gx_rw.append(np.concatenate([gx[arm], gx_rewind]))
            gy_rw.append(np.concatenate([gy[arm], gy_rewind]))

        kx = np.array(kx_rw)
        ky = np.array(ky_rw)
        gx = np.array(gx_rw)
        gy = np.array(gy_rw)
        t = np.arange(kx.shape[1]) * dt

    return kx, ky, gx, gy, t


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
        add_slew_limited_ramps=True
    ):
        """
        Initialize trajectory generator with imaging and hardware parameters.
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

    def _make_radius_profile(self):
        if self.add_slew_limited_ramps:
            ramp_up = self._slew_limited_ramp(self.ramp_samples)
            flat = np.ones(self.flat_samples)
            ramp_down = 1 - self._slew_limited_ramp(self.ramp_samples)
            r_profile = np.concatenate([ramp_up, flat, ramp_down])
        else:
            r_profile = np.ones(self.n_samples)
            r_profile[:self.ramp_samples] = np.linspace(0, 1, self.ramp_samples)
            r_profile[-self.ramp_samples:] = np.linspace(1, 0, self.ramp_samples)
        return r_profile

    def _enforce_gradient_limits(self, gx, gy):
        g_norm = np.sqrt(gx ** 2 + gy ** 2)
        over_gmax = g_norm > self.g_max
        if np.any(over_gmax):
            scale = self.g_max / np.max(g_norm)
            gx[over_gmax] *= scale
            gy[over_gmax] *= scale

        slew = np.sqrt(np.gradient(gx, self.dt) ** 2 + np.gradient(gy, self.dt) ** 2)
        over_smax = slew > self.s_max
        if np.any(over_smax):
            scale = self.s_max / np.max(slew)
            gx[over_smax] *= scale
            gy[over_smax] *= scale

        return gx, gy

    def _add_spoiler(self, kx, ky, gx, gy):
        n_spoil = self.ramp_samples
        spoil_area = 2 * self.k_max
        kx_out, ky_out, gx_out, gy_out = [], [], [], []
        for idx in range(self.n_interleaves):
            end_dir = np.array([gx[idx, -1], gy[idx, -1]])
            if np.linalg.norm(end_dir) == 0:
                end_dir = np.array([1, 0])
            else:
                end_dir /= np.linalg.norm(end_dir)
            g_spoil = end_dir * (spoil_area / (self.gamma * self.dt * n_spoil))
            kx_s = np.full(n_spoil, kx[idx, -1])
            ky_s = np.full(n_spoil, ky[idx, -1])
            gx_s = np.full(n_spoil, g_spoil[0])
            gy_s = np.full(n_spoil, g_spoil[1])
            kx_out.append(np.concatenate([kx[idx], kx_s]))
            ky_out.append(np.concatenate([ky[idx], ky_s]))
            gx_out.append(np.concatenate([gx[idx], gx_s]))
            gy_out.append(np.concatenate([gy[idx], gy_s]))
        return np.array(kx_out), np.array(ky_out), np.array(gx_out), np.array(gy_out)

    def _add_rewinder(self, kx, ky, gx, gy):
        n_rw = self.ramp_samples
        kx_out, ky_out, gx_out, gy_out = [], [], [], []
        for idx in range(self.n_interleaves):
            net_kx = kx[idx, -1]
            net_ky = ky[idx, -1]
            k_rewind = np.linspace([net_kx, net_ky], [0, 0], n_rw)
            gx_rewind = np.gradient(k_rewind[:, 0], self.dt) / self.gamma
            gy_rewind = np.gradient(k_rewind[:, 1], self.dt) / self.gamma
            kx_out.append(np.concatenate([kx[idx], k_rewind[:, 0]]))
            ky_out.append(np.concatenate([ky[idx], k_rewind[:, 1]]))
            gx_out.append(np.concatenate([gx[idx], gx_rewind]))
            gy_out.append(np.concatenate([gy[idx], gy_rewind]))
        return np.array(kx_out), np.array(ky_out), np.array(gx_out), np.array(gy_out)

    def generate(self):
        t = np.arange(self.n_samples) * self.dt
        kx = np.zeros((self.n_interleaves, self.n_samples))
        ky = np.zeros((self.n_interleaves, self.n_samples))
        gx = np.zeros((self.n_interleaves, self.n_samples))
        gy = np.zeros((self.n_interleaves, self.n_samples))

        r_profile = self._make_radius_profile()

        for idx in range(self.n_interleaves):
            if self.traj_type == 'spiral':
                phi = 2 * np.pi * idx / self.n_interleaves
                theta = self.turns * 2 * np.pi * t / t[-1] + phi
                r = r_profile * self.k_max
                kx[idx] = r * np.cos(theta)
                ky[idx] = r * np.sin(theta)
                gx[idx] = np.gradient(kx[idx], self.dt) / self.gamma
                gy[idx] = np.gradient(ky[idx], self.dt) / self.gamma
            elif self.traj_type == 'radial':
                angle = np.pi * idx / self.n_interleaves
                k_line = np.linspace(-self.k_max, self.k_max, self.n_samples)
                ramp = r_profile
                k_line = k_line * ramp
                kx[idx] = k_line * np.cos(angle)
                ky[idx] = k_line * np.sin(angle)
                gx[idx] = np.gradient(kx[idx], self.dt) / self.gamma
                gy[idx] = np.gradient(ky[idx], self.dt) / self.gamma
            else:
                raise ValueError("traj_type must be 'spiral' or 'radial'.")

            gx[idx], gy[idx] = self._enforce_gradient_limits(gx[idx], gy[idx])

        # Optionally add spoiler and/or rewinder
        if self.add_spoiler:
            kx, ky, gx, gy = self._add_spoiler(kx, ky, gx, gy)
        if self.add_rewinder:
            kx, ky, gx, gy = self._add_rewinder(kx, ky, gx, gy)

        t = np.arange(kx.shape[1]) * self.dt
        return kx, ky, gx, gy, t

    def check_gradient_and_slew_limits(self, k_traj):
        """
        Check limits for a single interleaf/spoke trajectory
        """
        gamma = self.gamma
        G = np.diff(k_traj, axis=0) / self.dt / gamma
        slew = np.diff(G, axis=0) / self.dt
        grad_ok = np.all(np.abs(G) <= self.g_max)
        slew_ok = np.all(np.abs(slew) <= self.s_max)
        return grad_ok, slew_ok, G, slew

# Example usage:
# gen = KSpaceTrajectoryGenerator(
#     fov=0.24, resolution=0.001, dt=4e-6, g_max=40e-3, s_max=150.0,
#     n_interleaves=8, traj_type='spiral', ramp_fraction=0.1,
#     add_rewinder=True, add_spoiler=True, add_slew_limited_ramps=True
# )
# kx, ky, gx, gy, t = gen.generate()

def gradient_and_pns_check(k_traj, grad_max, slew_max, dt):
    """
    k_traj: [N, D] array, N points, D dimensions
    grad_max: T/m
    slew_max: T/m/s
    dt: time between points (s)
    """
    gamma = 42.58e6  # Hz/T
    G = np.diff(k_traj, axis=0) / dt / gamma
    slew = np.diff(G, axis=0) / dt

    grad_ok = np.all(np.abs(G) <= grad_max)
    slew_ok = np.all(np.abs(slew) <= slew_max)
    return grad_ok, slew_ok, G, slew
