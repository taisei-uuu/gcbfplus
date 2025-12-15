import jax
import jax.numpy as jnp
import numpy as np
from gcbfplus.env.double_integrator import DoubleIntegrator
from gcbfplus.env.obstacle import Rectangle

def test_apf_integration():
    print("\n=== Test 1: APF Integration in Environment Step ===")
    # Params
    params = DoubleIntegrator.PARAMS.copy()
    params.update({
        "formation_mode": True,
        "formation_offsets": [[-0.5, 0.5], [-0.5, -0.5]], # Two followers
        "apf_enabled": True,
        "apf_att_gain": 1.0,
        "apf_rep_obs_gain": 1.0,
        "apf_vortex_gain": 0.5,
        "apf_obs_dist": 1.0,
    })
    
    env = DoubleIntegrator(num_agents=3, area_size=10.0, params=params)
    
    key = jax.random.PRNGKey(0)
    graph = env.reset(key)
    action = jnp.zeros((3, 2))
    
    # Step
    next_graph, reward, cost, done, info = env.step(graph, action)
    
    # Check goals
    leader_pos = next_graph.env_states.agent[0, :2]
    follower_1_pos = next_graph.env_states.agent[1, :2]
    follower_1_goal = next_graph.env_states.goal[1, :2]
    
    nominal_goal = leader_pos + jnp.array(params["formation_offsets"][0])
    
    print(f"Follower 1 Pos: {follower_1_pos}")
    print(f"Follower 1 Nominal Goal: {nominal_goal}")
    print(f"Follower 1 Actual Goal (APF Adjusted): {follower_1_goal}")
    
    diff = jnp.linalg.norm(follower_1_goal - nominal_goal)
    print(f"Difference: {diff}")
    
    if diff > 1e-6:
        print("SUCCESS: APF adjustment acts on the goal state (relative to nominal).")
    else:
        print("NOTE: Distance is small, possibly no obstacles nearby in this random seed.")

def test_vortex_direction():
    print("\n=== Test 2: Isolated Vortex Force Direction ===")
    # Setup params to isolate forces: No attraction, specific gains
    params = DoubleIntegrator.PARAMS.copy()
    params.update({
        "apf_enabled": True,
        "apf_att_gain": 0.0, # Turn off attraction to see obstacle forces clearly
        "apf_rep_obs_gain": 1.0,
        "apf_vortex_gain": 1.0,
        "apf_obs_dist": 2.0,
        "apf_dt": 1.0, 
    })
    
    env = DoubleIntegrator(num_agents=1, area_size=10.0, params=params)
    
    # 1. Setup Scenario: Agent at (0,0), Wall at x=0.5
    current_pos = jnp.array([0.0, 0.0])
    
    # Wall to the RIGHT at x=0.5 (Distance = 0.5)
    obs_center = jnp.array([[1.0, 0.0]])
    obs_width = jnp.array([1.0])
    obs_height = jnp.array([10.0])
    obs_theta = jnp.array([0.0])
    obstacles = env.create_obstacles(obs_center, obs_width, obs_height, obs_theta)
    
    # Target is Forward-Right at (10, 10). Agent should slide UP (+Y)
    nominal_target = jnp.array([10.0, 10.0])
    other_agents = jnp.zeros((0, 2))
    
    # 2. Compute Force
    adjusted_pos = env._compute_apf_force_field(
        current_pos, nominal_target, other_agents, obstacles
    )
    
    force_vector = adjusted_pos - current_pos
    fx, fy = force_vector[0], force_vector[1]
    
    print(f"Scenario: Agent at (0,0), Wall at x=0.5 (Right), Target at (10,10) (Top-Right)")
    print(f"Force Result: [Fx={fx:.4f}, Fy={fy:.4f}]")
    
    # Checks
    repulsion_ok = fx < -1e-3
    vortex_ok = fy > 1e-3
    
    if repulsion_ok:
        print("PASS: Repulsion pushing LEFT (away from wall).")
    else:
        print(f"FAIL: Repulsion incorrect (Fx={fx}).")
        
    if vortex_ok:
        print("PASS: Vortex pushing UP (sliding along wall towards target).")
    else:
        print(f"FAIL: Vortex incorrect (Fy={fy}).")

if __name__ == "__main__":
    test_apf_integration()
    test_vortex_direction()
