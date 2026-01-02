import jax
import jax.numpy as jnp
import numpy as np
import pathlib
from gcbfplus.env import DoubleIntegrator
from gcbfplus.env.base import RolloutResult

def test_inspection_target():
    print("Testing Inspection Target Logic...")
    
    # Config: 
    # Obstacle 1: Inspection Target (Agent 0 should ignore)
    # Obstacle 2: Normal Obstacle (Agent 0 should avoid, Agent 1 should avoid both)
    
    fixed_config = {
        "obstacles": [
            {
                "shape": "rectangle",
                "pos": [0.0, 0.0],
                "width": 1.0,
                "height": 1.0,
                "velocity": [0.0, 0.0],
                "inspection_target": True # TARGET
            },
            {
                "shape": "rectangle",
                "pos": [2.0, 2.0],
                "width": 1.0,
                "height": 1.0,
                "velocity": [0.0, 0.0],
                "inspection_target": False # NORMAL
            }
        ],
        "agents": {
            # Agent 0 start inside Inspection Target
            # Agent 1 start inside Inspection Target (Collision!)
            "start": [[0.0, 0.0], [0.0, 0.0]], 
            "goal": [[1.0, 1.0], [3.0, 3.0]]
        }
    }
    
    params = DoubleIntegrator.PARAMS.copy()
    params.update({
        "fixed_config": fixed_config,
        "n_obs": 2,
        "num_agents": 2,
        "virtual_leader": False # Disable virtual leader to test inspection logic independently
    })
    
    env = DoubleIntegrator(num_agents=2, area_size=4.0, params=params)
    key = jax.random.PRNGKey(0)
    
    print("Resetting environment...")
    # Note: Reset might fail for Agent 1 because it's inside obstacle, 
    # but `get_node_goal_rng` is skipped for fixed_config.
    # So it should succeed.
    
    graph = env.reset(key)
    print("Reset successful!")
        
    obstacles = graph.env_states.obstacle
    agent_pos = graph.states[:2, :2]
    
    print(f"Agent 0 Pos: {agent_pos[0]}")
    print(f"Agent 1 Pos: {agent_pos[1]}")
    
    # Check Safety Masks
    # Agent 0 should be SAFE regarding Obs 1 (Inspection), UNSAFE regarding Obs 2 (if close)
    # Agent 1 should be UNSAFE regarding Obs 1
    
    # safe_mask returns (N_agent,)
    # It combines SAFE_AGENT and SAFE_OBS
    # We want to check SAFE_OBS component effectively.
    
    unsafe_mask = env.unsafe_mask(graph)
    print(f"Unsafe Mask (should be [False, True]): {unsafe_mask}")
    
    # Verify Agent 0 is safe (False in unsafe_mask) because it's inside inspection target
    # Agent 1 is unsafe (True) because it's inside inspection target (normal for it)
    
    assert unsafe_mask[0] == False, "Agent 0 should be safe inside inspection target!"
    assert unsafe_mask[1] == True, "Agent 1 should be unsafe inside inspection target!"
    
    print("Assertion Passed: Agent 0 ignores inspection target collision.")
    
    # Visualization
    rollout_len = 5
    Tp1_graph = jax.tree_util.tree_map(lambda x: jnp.stack([x]*(rollout_len+1)), graph)
    T_action = jnp.zeros((rollout_len, 2, 2))
    T_reward = jnp.zeros((rollout_len,))
    T_cost = jnp.zeros((rollout_len,))
    T_done = jnp.zeros((rollout_len,), dtype=bool)
    T_info = jax.tree_util.tree_map(lambda x: jnp.stack([x]*rollout_len), {})
    
    rollout = RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info)
    Ta_is_unsafe = jnp.zeros((rollout_len, 2), dtype=bool)
    video_path = pathlib.Path("test_inspection.mp4")
    
    print("rendering video...")
    env.render_video(rollout, video_path, Ta_is_unsafe=Ta_is_unsafe)
    print(f"Video saved to {video_path}")

if __name__ == "__main__":
    test_inspection_target()
