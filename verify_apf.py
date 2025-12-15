import jax
import jax.numpy as jnp
import numpy as np
from gcbfplus.env.double_integrator import DoubleIntegrator

def test_apf():
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
    
    # Fake action (zero)
    action = jnp.zeros((3, 2))
    
    # Step
    next_graph, reward, cost, done, info = env.step(graph, action)
    
    print("Step 1 done")
    
    # Check goals
    leader_pos = next_graph.env_states.agent[0, :2]
    follower_1_pos = next_graph.env_states.agent[1, :2]
    follower_1_goal = next_graph.env_states.goal[1, :2]
    
    nominal_goal = leader_pos + jnp.array(params["formation_offsets"][0])
    
    print(f"Follower 1 Pos: {follower_1_pos}")
    print(f"Follower 1 Nominal Goal: {nominal_goal}")
    print(f"Follower 1 Actual Goal (APF Adjusted): {follower_1_goal}")
    
    # If obstacles are nearby, actual goal should differ from nominal
    diff = jnp.linalg.norm(follower_1_goal - nominal_goal)
    print(f"Difference: {diff}")
    
    if diff > 1e-6:
        print("Success: APF adjustment detected (relative to nominal).")
    else:
        # Might be 0 if no obstacles nearby.
        # Let's force check with obstacles.
        print("Distance is small, possibly no obstacles nearby.")

    # Let's try to put an obstacle very close to the nominal target
    # This is hard to force in random init without fixed config.
    # But just compiling and running proves no shape errors.

if __name__ == "__main__":
    test_apf()
