
import jax
import jax.numpy as jnp
import numpy as np
from gcbfplus.env.double_integrator import DoubleIntegrator

def test_backstepping_implementation():
    print("Testing Backstepping Implementation...")
    
    # Setup Environment
    num_agents = 3
    params = {
        "formation_mode": True,
        "formation_offsets": [[-0.5, 0.0], [0.5, 0.0]], # 2 followers
        "kp_bs": 1.0,
        "kv_bs": 2.0,
        "m": 1.0 # simplify mass
    }
    env = DoubleIntegrator(num_agents=num_agents, area_size=10.0, params=params)
    key = jax.random.PRNGKey(0)
    
    # Reset Environment
    graph = env.reset(key)
    
    # 1. Verify u_ref matches manual backstepping calculation
    print("\n[Test 1] Verifying u_ref calculation...")
    
    agent_states = graph.type_states(type_idx=0, n_type=num_agents)
    goal_states = graph.type_states(type_idx=1, n_type=num_agents)
    
    x_i = agent_states[:, :2]
    v_i = agent_states[:, 2:]
    p_d = goal_states[:, :2]
    v_d = goal_states[:, 2:]
    
    kp = params["kp_bs"]
    kv = params["kv_bs"]
    
    # Manual Calculation
    e_p = x_i - p_d
    v_ref = -kp * e_p + v_d
    e_v = v_i - v_ref
    u_should_be = -kv * e_v - e_p
    
    # Action from u_ref
    u_actual = env.u_ref(graph)
    
    # Compare (allowing for small differences due to clipping/float precision)
    diff = jnp.linalg.norm(u_should_be - u_actual)
    print(f"Difference between manual and implemented u_ref: {diff}")
    
    if diff < 1e-5:
        print(">>> u_ref verification PASSED")
    else:
        print(">>> u_ref verification FAILED")
        print("Expected:\n", u_should_be)
        print("Actual:\n", u_actual)

    # 2. Verify Follower Goal Velocity Update
    print("\n[Test 2] Verifying Follower Goal Velocity Update...")
    
    # Perform a step
    # We give some action to move the agents
    action = jnp.zeros((num_agents, 2))
    # Give leader some velocity
    # agent_states is [x, y, vx, vy]
    # We can rely on 'step' to integrate. 
    # But wait, step uses action to update velocity. 
    # Let's apply an action that accelerates the leader.
    action = action.at[0].set(jnp.array([1.0, 0.5])) 
    
    next_graph, _, _, _, _ = env.step(graph, action)
    
    # Check next state
    next_agent_states = next_graph.type_states(type_idx=0, n_type=num_agents)
    next_goal_states = next_graph.type_states(type_idx=1, n_type=num_agents)
    
    leader_vel = next_agent_states[0, 2:]
    follower_goal_vels = next_goal_states[1:, 2:]
    
    print(f"Leader Velocity: {leader_vel}")
    print(f"Follower Goal Velocities:\n{follower_goal_vels}")
    
    # Check if follower goal velocities match leader velocity
    vel_diff = jnp.linalg.norm(follower_goal_vels - leader_vel, axis=1)
    max_vel_diff = jnp.max(vel_diff)
    print(f"Max velocity difference: {max_vel_diff}")
    
    if max_vel_diff < 1e-5:
        print(">>> Goal Velocity Update PASSED")
    else:
        print(">>> Goal Velocity Update FAILED")

if __name__ == "__main__":
    test_backstepping_implementation()
