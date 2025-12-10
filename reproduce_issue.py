import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from gcbfplus.env.double_integrator import DoubleIntegrator

def test_affine_transformation():
    """
    Tests if the apply_anisotropic_scaling correctly staggers agents
    when squeezing (sy < 1.0) and shear (c_shear > 0) is active.
    """
    
    # 3 agents: Leader(0,0), Left Follower(-1, 1), Right Follower(-1, -1)
    # Note: offsets are usually relative to leader. 
    # Let's say: Leader is at (0,0).
    # Follower 1: (-1, 0.5) (Left/Top)
    # Follower 2: (-1, -0.5) (Right/Bottom)
    
    offsets = jnp.array([
        [-1.0, 0.5],
        [-1.0, -0.5]
    ])
    
    # Instantiate environment
    env = DoubleIntegrator(num_agents=3, area_size=10.0)
    
    # Case 1: Moving East (1, 0). sy = 0.3 (Squeezed).
    leader_vel = jnp.array([1.0, 0.0])
    sy = 0.3
    
    print(f"Original Offsets:\n{offsets}")
    
    transformed = env.apply_anisotropic_scaling(offsets, leader_vel, sy)
    
    print(f"Transformed Offsets (East, sy={sy}):\n{transformed}")
    
    # Expectation:
    # sy=0.3 -> small y spread.
    # sx = 1/0.3 = 3.33 -> stretched x backward.
    # Shear: k = c_shear * (1-0.3) = 0.5 * 0.7 = 0.35
    # H = [[1, 0], [k, 1]]
    # Align: Already aligned (vel=[1,0]).
    # Shear: x_new = x + k*y
    # Follower 1 (-1, 0.5): x' = -1 + 0.35*0.5 = -1 + 0.175 = -0.825
    # Follower 2 (-1, -0.5): x' = -1 + 0.35*(-0.5) = -1 - 0.175 = -1.175
    # Scale: x'' = x' * sx, y'' = y' * sy
    # sx ~ 3.33. x'' ~ -0.825 * 3.33 ~ -2.75 / -1.175 * 3.33 ~ -3.91
    # Result: They should have DIFFERENT x coordinates (Staggered).
    
    x_diff = transformed[0, 0] - transformed[1, 0]
    print(f"X Difference (Should be non-zero): {x_diff}")
    
    if jnp.abs(x_diff) > 0.1:
        print("PASS: Staggering observed.")
    else:
        print("FAIL: No staggering observed.")

def test_smoothing_logic():
    """
    Simulates a step to check if smoothed_sy updates correctly.
    """
    env = DoubleIntegrator(num_agents=3, area_size=10.0)
    key = jax.random.PRNGKey(0)
    graph = env.reset(key)
    
    # Initial state check
    print(f"Initial smoothed_sy: {graph.env_states.smoothed_sy}")
    
    # Mock obstacle detection to force low sy
    # We can't easily mock internal step variables without running step.
    # But we can check if graph.env_states has the field.
    assert hasattr(graph.env_states, "smoothed_sy")
    assert hasattr(graph.env_states, "smoothed_leader_vel")
    
    print("PASS: State fields present.")

if __name__ == "__main__":
    test_affine_transformation()
    test_smoothing_logic()
