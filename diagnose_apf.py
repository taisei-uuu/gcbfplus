
import jax
import jax.numpy as jnp
import numpy as np
from gcbfplus.env.double_integrator import DoubleIntegrator
from gcbfplus.env.obstacle import Rectangle

def diagnose():
    print("Diagnosing APF Broadcasting Error...")
    
    # 1. Setup Environment similar to user scenario
    n_followers = 6
    num_agents = n_followers + 1
    n_obs = 4
    n_rays = 32
    
    params = {
        "n_obs": n_obs,
        "n_rays": n_rays,
        "obstacle_type": "rectangle",
        "apf_enabled": True, # ERROR TRIGGER
        "formation_mode": True,
        "formation_offsets": jnp.zeros((n_followers + 1, 2)), # Dummy zeros
        "formation_flexible_assignment": True,
    }
    
    env = DoubleIntegrator(num_agents=num_agents, area_size=10.0, params=params)
    
    # 2. Create Dummy State
    key = jax.random.PRNGKey(0)
    
    # Create valid Rectangle obstacles (4 of them)
    # Rectangle fields: center(2), width, height, theta, velocity(2) ...
    # And points(4, 2)
    
    # To properly init obstacles, let's just use reset (it creates obstacles)
    # But mixed static/dynamic requires my recent change code.
    # Let's trust reset to make valid obstacles.
    graph = env.reset(key)
    obstacles = graph.env_states.obstacle
    
    # 3. Prepare Inputs for _apply_apf_adjustment
    leader_pos = jnp.array([2.0, 5.0])
    follower_positions = jnp.zeros((n_followers, 2)) + jnp.array([2.0, 5.0]) # Stacked on leader
    nominal_offsets = jnp.zeros((n_followers, 2))
    
    print(f"Obstacle Type: {type(obstacles)}")
    print(f"Obstacle Points Shape: {obstacles.points.shape} (Expected: ({n_obs}, 4, 2))")
    
    # 4. Run _apply_apf_adjustment and see if it crashes
    print("\nRunning _apply_apf_adjustment...")
    try:
        # Wrap in JIT to simulate "jit rollout" constraint which might trigger shape checks
        @jax.jit
        def test_fn(l_pos, f_pos, offsets, obs):
            return env._apply_apf_adjustment(l_pos, f_pos, offsets, obs)

        adj_targets = test_fn(leader_pos, follower_positions, nominal_offsets, obstacles)
        print("Success! Output shape:", adj_targets.shape)
        
    except Exception as e:
        print("\n!!! CAUGHT EXCEPTION !!!")
        print(e)
        
        # If JIT failed, try running non-JIT to get better traceback or see if it passes
        # The user error was in JIT comp.
        print("\nAttempting non-JIT run to verify logic...")
        try:
             env._apply_apf_adjustment(leader_pos, follower_positions, nominal_offsets, obstacles)
             print("Non-JIT run passed (Error is specific to JIT/Broadcasting compilation).")
        except Exception as e2:
             print("Non-JIT also failed:", e2)

if __name__ == "__main__":
    diagnose()
