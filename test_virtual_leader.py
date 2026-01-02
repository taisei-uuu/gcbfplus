import jax
import jax.numpy as jnp
import numpy as np
import pathlib
from gcbfplus.env import DoubleIntegrator
from gcbfplus.env.base import RolloutResult

def test_virtual_leader():
    print("Testing Virtual Leader Mode...")
    
    # Config: Agent 0 (Virtual Leader) inside an obstacle
    fixed_config = {
        "obstacles": [
            {
                "shape": "rectangle",
                "pos": [0.0, 0.0],
                "width": 1.0,
                "height": 1.0,
                "velocity": [0.0, 0.0]
            }
        ],
        "agents": {
            # Agent 0 start inside the rectangle at [0,0]
            "start": [[0.0, 0.0], [2.0, 2.0]], 
            "goal": [[1.0, 1.0], [3.0, 3.0]]
        }
    }
    
    params = DoubleIntegrator.PARAMS.copy()
    params.update({
        "fixed_config": fixed_config,
        "n_obs": 1,
        "num_agents": 2,
        "virtual_leader": True # ENABLE VIRTUAL LEADER
    })
    
    env = DoubleIntegrator(num_agents=2, area_size=4.0, params=params)
    key = jax.random.PRNGKey(0)
    
    print("Resetting environment (should succeed despite Agent 0 being inside obstacle)...")
    try:
        graph = env.reset(key)
        print("Reset successful!")
    except Exception as e:
        print(f"Reset failed: {e}")
        raise e
        
    obstacles = graph.env_states.obstacle
    agent_pos = graph.states[:2, :2]
    
    print(f"Agent 0 Pos: {agent_pos[0]}")
    print(f"Obstacle Center: {obstacles.center[0]}")
    
    # Verify cost calculation
    # Agent 0 is inside obstacle, so collision would be True if not virtual leader
    # But virtual leader should ignore it.
    # Agent 1 is outside
    
    # We can check `get_cost` indirectly or check `safe_mask`
    # Let's check `safe_mask`
    
    safe_mask = env.safe_mask(graph)
    # safe_mask is boolean array? No, it's typically binary 0/1 or bool
    # safe_mask method returns array.
    
    print(f"Safe Mask: {safe_mask}")
    
    # Logic in `safe_mask`: 
    # if virtual_leader: safe_obs[0] = True
    # safe_mask = safe_agent & safe_obs
    
    # Since we just reset, agents might be close to each other or fine.
    # Agent 0 is definitely inside obstacle. If virtual_leader logic works, safe_obs[0] should be True.
    # If not working, safe_obs[0] would be False.
    
    # Let's inspect `safe_mask` values
    # safe_mask is (n_agent, 1) or (n_agent,) ?
    # It returns (N, 1) usually or similar.
    
    # If safe_mask for agent 0 is True (1.0), strict success.
    
    # For visualization check, we simulate a small rollout
    rollout_len = 5
    Tp1_graph = jax.tree_util.tree_map(lambda x: jnp.stack([x]*(rollout_len+1)), graph)
    T_action = jnp.zeros((rollout_len, 2, 2))
    T_reward = jnp.zeros((rollout_len,))
    T_cost = jnp.zeros((rollout_len,))
    T_done = jnp.zeros((rollout_len,), dtype=bool)
    T_info = jax.tree_util.tree_map(lambda x: jnp.stack([x]*rollout_len), {})
    
    rollout = RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info)
    
    # Ta_is_unsafe for viz
    Ta_is_unsafe = jnp.zeros((rollout_len, 2), dtype=bool)
    
    video_path = pathlib.Path("test_virtual_leader.mp4")
    
    print("rendering video...")
    env.render_video(rollout, video_path, Ta_is_unsafe=Ta_is_unsafe)
    print(f"Video saved to {video_path}")

if __name__ == "__main__":
    test_virtual_leader()
