
import jax
import jax.numpy as jnp
import numpy as np
from gcbfplus.env.double_integrator import DoubleIntegrator as DI

def test_scaling_logic():
    # User's new params
    params = {
        "m": 0.1,
        "comm_radius": 0.5,
        "s_min": 0.3,
        "d_critical": 0.3,
        "d_free": 0.6,
        "formation_mode": True,
        "formation_offsets": [[-0.5, 0.5], [-0.5, -0.5]], # Triangle followers
        "n_rays": 32,
        "car_radius": 0.05,
        "n_obs": 0,
        "obs_len_range": [0.2, 0.2],
    }

    env = DI(num_agents=3, area_size=10.0, params=params)

    print("=== Testing Scaling Logic with User Params ===")
    print(f"Params: d_free={params['d_free']}, d_critical={params['d_critical']}, comm_radius={params['comm_radius']}")

    # Case 1: No Obstacle (Distance = comm_radius = 0.5)
    # Rationale: Lidar returns max range (normalized 1.0) -> 0.5m
    dist_1 = 0.5
    sy_1 = env.compute_scaling_factor_y(dist_1)
    print(f"\n[Case 1] Max Sensing Distance ({dist_1}m):")
    print(f"  sy: {sy_1:.4f}")
    
    leader_vel = jnp.array([1.0, 0.0])
    offsets = jnp.array(params["formation_offsets"])
    scaled_offsets_1 = env.apply_anisotropic_scaling(offsets, leader_vel, sy_1)
    print(f"  Original: {offsets[0]}")
    print(f"  Scaled:   {scaled_offsets_1[0]}")

    # Case 2: Obstacle Detected (Distance = 0.2)
    dist_2 = 0.2
    sy_2 = env.compute_scaling_factor_y(dist_2)
    print(f"\n[Case 2] Obstacle Detected ({dist_2}m):")
    print(f"  sy: {sy_2:.4f}")
    
    scaled_offsets_2 = env.apply_anisotropic_scaling(offsets, leader_vel, sy_2)
    print(f"  Scaled:   {scaled_offsets_2[0]}")

    # Check difference
    diff_y = jnp.abs(scaled_offsets_1[0, 1] - scaled_offsets_2[0, 1])
    diff_x = jnp.abs(scaled_offsets_1[0, 0] - scaled_offsets_2[0, 0])
    
    print(f"\n[Comparison]")
    print(f"  Change in Y: {diff_y:.4f}")
    print(f"  Change in X: {diff_x:.4f}")
    
    if diff_y < 1e-3:
        print("WARNING: No significant change in Y detected!")
    else:
        print("SUCCESS: Scaling is changing.")

if __name__ == "__main__":
    test_scaling_logic()
