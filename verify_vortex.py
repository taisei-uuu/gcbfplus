import jax
import jax.numpy as jnp
import numpy as np
from gcbfplus.env.double_integrator import DoubleIntegrator
from gcbfplus.env.obstacle import Rectangle

def test_vortex_direction():
    # Setup params
    params = DoubleIntegrator.PARAMS.copy()
    params.update({
        "apf_enabled": True,
        "apf_att_gain": 0.0, # Turn off attraction to isolate obstacle forces
        "apf_rep_obs_gain": 1.0,
        "apf_vortex_gain": 1.0,
        "apf_obs_dist": 2.0, # Large enough to see effect
        "apf_dt": 1.0, # Large dt to see result as force vector directly
    })
    
    env = DoubleIntegrator(num_agents=1, area_size=10.0, params=params)
    
    # 1. Setup Scenario
    # Agent at Origin (0,0)
    current_pos = jnp.array([0.0, 0.0])
    
    # Wall to the RIGHT at x=0.5 (Distance = 0.5)
    # Rectangle center at (1.0, 0.0), width=1.0, height=10.0
    # Left edge is at 1.0 - 0.5 = 0.5.
    obs_center = jnp.array([[1.0, 0.0]])
    obs_width = jnp.array([1.0])
    obs_height = jnp.array([10.0])
    obs_theta = jnp.array([0.0])
    obstacles = env.create_obstacles(obs_center, obs_width, obs_height, obs_theta)
    
    # Target is Forward-Right at (10, 10)
    # This means the agent "should" slide UP (positive Y) along the wall to get closer to y=10.
    nominal_target = jnp.array([10.0, 10.0])
    
    # Other agents (none)
    other_agents = jnp.zeros((0, 2))
    
    # 2. Compute Adjusted Position directly
    # adjusted = current + (F_rep + F_vortex) * dt
    # Since we set att_gain=0 and dt=1, adjusted - current = F_rep + F_vortex
    
    adjusted_pos = env._compute_apf_force_field(
        current_pos, nominal_target, other_agents, obstacles
    )
    
    force_vector = adjusted_pos - current_pos
    
    print(f"Current Pos: {current_pos}")
    print(f"Obstacle Wall at x=0.5")
    print(f"Target: {nominal_target}")
    print(f"Resulting Force Vector: {force_vector}")
    
    fx, fy = force_vector[0], force_vector[1]
    
    print(f"Fx (Repulsion expected negative): {fx}")
    print(f"Fy (Vortex expected positive): {fy}")
    
    # Checks
    if fx < -1e-3:
        print("PASS: Repulsion pushing away from wall (Left).")
    else:
        print("FAIL: No repulsive force away from wall.")
        
    if fy > 1e-3:
        print("PASS: Vortex pushing along wall (Up towards target).")
    elif fy < -1e-3:
        print("FAIL: Vortex pushing WRONG way (Down).")
    else:
        print("FAIL: No vortex force.")

if __name__ == "__main__":
    test_vortex_direction()
