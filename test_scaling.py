
import jax
import jax.numpy as jnp
import numpy as np
import jax.random as jr
from gcbfplus.env import make_env
import matplotlib.pyplot as plt

def test_scaling():
    print("Testing Anisotropic Scaling...")
    
    # 1. Initialize environment with formation mode
    num_agents = 3 # 1 leader + 2 followers
    # Offsets: Follower 1 at (0, 0.5), Follower 2 at (0, -0.5)
    formation_offsets = [[0.0, 0.5], [0.0, -0.5]]
    
    env = make_env(
        env_id="DoubleIntegrator",
        num_agents=num_agents,
        num_obs=0, # We will inject obstacles manually or rely on random gen affecting scale
        area_size=10.0,
        formation_mode=True,
        formation_offsets=formation_offsets
    )
    
    # 2. Setup a state where agents are close to an obstacle
    # We can inject a specific state or just check the scaling methods directly since we exposed them
    # But checking via step() is better integration test.
    
    # Let's create a dummy graph/state
    key = jr.PRNGKey(0)
    graph = env.reset(key)
    
    # Inject obstacles close to the agents
    # Agent 0 (Goal) is at some position, let's say (0,0)
    # Put obstacle at (0, 1.0) -> distance 1.0. d_critical=1.2. So scaling should trigger.
    
    # Modify env_states manually? The env_states is inside the graph.
    # graph.env_states.obstacle is the obstacle object.
    # Obstacles are parameterized by pos, len_x, len_y, theta
    
    # Let's verify the helper function logic first (Unit Test style)
    print("\n--- Unit Test: Helper Methods ---")
    
    d_vals = jnp.array([0.5, 1.0, 1.2, 1.8, 2.4, 3.0])
    s_mins = env._params["s_min"]
    d_crit = env._params["d_critical"]
    d_free = env._params["d_free"]
    
    print(f"Params: s_min={s_mins}, d_crit={d_crit}, d_free={d_free}")
    
    sy_vals = env.compute_scaling_factor_y(d_vals)
    sx_vals = env.compute_scaling_factor_x(sy_vals)
    
    print("Distances:", d_vals)
    print("Sy values:", sy_vals)
    print("Sx values:", sx_vals)
    print("Area (Sx*Sy):", sx_vals * sy_vals)
    
    assert jnp.allclose(sx_vals * sy_vals, 1.0), "Area preservation failed!"
    assert sy_vals[0] == s_mins, "Sy should be s_min below d_critical"
    assert sy_vals[-1] == 1.0, "Sy should be 1.0 above d_free"
    
    # 3. Integration Test: offset scaling
    print("\n--- Integration Test: Offset Scaling ---")
    offsets = jnp.array(formation_offsets)
    leader_vel = jnp.array([1.0, 0.0]) # Moving in X direction
    
    # Case 1: No scaling (simulate large distance)
    sy_free = 1.0
    scaled_offsets_free = env.apply_anisotropic_scaling(offsets, leader_vel, sy_free)
    print("Original Offsets:\n", offsets)
    print("Scaled (Free):\n", scaled_offsets_free)
    assert jnp.allclose(offsets, scaled_offsets_free), "Offsets shouldn't change when free"
    
    # Case 2: Max scaling (sy = s_min)
    sy_min = s_mins
    scaled_offsets_min = env.apply_anisotropic_scaling(offsets, leader_vel, sy_min)
    print(f"Scaled (Min, sy={sy_min}):\n", scaled_offsets_min)
    
    # With leader vel = (1,0), rotation is 0.
    # x should scale by 1/s_min, y should scale by s_min.
    # Original: (0, 0.5), (0, -0.5)
    # Expected: (0, 0.5*s_min), (0, -0.5*s_min)
    expected_y = offsets[:, 1] * sy_min
    expected_x = offsets[:, 0] * (1.0/sy_min) # Should be 0 * something = 0
    
    assert jnp.allclose(scaled_offsets_min[:, 1], expected_y), "Y scaling incorrect"
    assert jnp.allclose(scaled_offsets_min[:, 0], expected_x), "X scaling incorrect"
    
    # Case 3: Rotated Leader Velocity
    leader_vel_rot = jnp.array([0.0, 1.0]) # Moving in Y direction
    # Rotation angle = 90 deg (pi/2)
    # R(90) = [[0, -1], [1, 0]]
    # Original P (rows): (0, 0.5) -> vector p = [0, 0.5]^T
    # 1. Rotate back (-90): R(-90) = [[0, 1], [-1, 0]]. 
    #    p' = [[0, 1], [-1, 0]] @ [0, 0.5] = [0.5, 0] (Now it's on X axis!)
    # 2. Scale: S = diag(sx, sy). 
    #    p'' = [sx * 0.5, sy * 0] = [sx * 0.5, 0]
    # 3. Rotate forward (90): R(90) = [[0, -1], [1, 0]]
    #    p_final = [[0, -1], [1, 0]] @ [sx * 0.5, 0] = [0, sx * 0.5]
    
    # So effectively, the offset (0, 0.5) which was ALONG the motion direction (since motion is Y),
    # got scaled by sx (expanded).
    
    scaled_offsets_rot = env.apply_anisotropic_scaling(offsets, leader_vel_rot, sy_min)
    print("Leader Vel (0, 1) -> Moving Y")
    print("Original Offsets (Transverse to X, Parallel to Y):\n", offsets)
    print("Scaled (Rotated Leader):\n", scaled_offsets_rot)
    
    # Offset 0: (0, 0.5). It aligns with leader velocity. Should assume value (0, 0.5 * sx)
    # sx = 1.0/0.7 approx 1.428
    sx = 1.0 / sy_min
    assert jnp.allclose(scaled_offsets_rot[0, 1], 0.5 * sx), "Parallel component should expand"
    assert jnp.allclose(scaled_offsets_rot[0, 0], 0.0, atol=1e-5), "Perpendicular component should remain 0"

    print("\nTests Passed!")

if __name__ == "__main__":
    test_scaling()
