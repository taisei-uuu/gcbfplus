import jax
import jax.numpy as jnp
import numpy as np
import pathlib
from gcbfplus.env import DoubleIntegrator
from gcbfplus.env.base import RolloutResult
from gcbfplus.utils.graph import GraphsTuple

def test_mixed_viz():
    print("Testing Mixed Obstacle Visualization...")
    
    fixed_config = {
        "obstacles": [
            {
                "shape": "circle",
                "pos": [0.5, 0.5],
                "r": 0.2,
                "velocity": [0.0, 0.0]
            },
            {
                "shape": "rectangle",
                "pos": [-0.5, -0.5],
                "width": 0.2,
                "height": 0.4,
                "theta": 0.0,
                "velocity": [0.1, 0.0]
            }
        ],
        "agents": {
            "start": [[0.0, 0.0]], 
            "goal": [[1.0, 1.0]]
        }
    }
    
    params = DoubleIntegrator.PARAMS.copy()
    params.update({
        "fixed_config": fixed_config,
        "n_obs": 2,
        "num_agents": 1
    })
    
    env = DoubleIntegrator(num_agents=1, area_size=2.0, params=params)
    key = jax.random.PRNGKey(0)
    graph0 = env.reset(key)
    
    # Create a dummy rollout result
    # We need a list of graphs, but RolloutResult expects Tp1_graph to be stacked graphs
    # Let's just stack graph0 twice for T=1
    
    # We need to use tree_stack from utils, or just rely on the fact that graph is a NamedTuple of arrays
    # But wait, RolloutResult.Tp1_graph expects a GraphsTuple where leaves have shape (T+1, ...)
    
    def stack_leaves(leaves):
        return jnp.stack([leaves, leaves], axis=0)
    
    Tp1_graph = jax.tree_util.tree_map(stack_leaves, graph0)
    
    # Dummy other fields
    T_action = jnp.zeros((1, 1, 2))
    T_reward = jnp.zeros((1,))
    T_cost = jnp.zeros((1,))
    T_done = jnp.zeros((1,), dtype=bool)
    T_info = jax.tree_util.tree_map(lambda x: jnp.stack([x], axis=0), {}) # Empty info
    
    # We need to match the structure expected by render_video loop
    # render_video iterates over range(len(T_graph.n_node))
    # and calls tree_index(T_graph, kk)
    
    rollout = RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info)
    
    video_path = pathlib.Path("test_mixed_viz.mp4")
    
    # Ta_is_unsafe needs to be list or array of shape (T, n_agents)
    Ta_is_unsafe = jnp.zeros((1, 1), dtype=bool)
    
    try:
        env.render_video(rollout, video_path, Ta_is_unsafe=Ta_is_unsafe)
        print("render_video executed successfully.")
    except Exception as e:
        print(f"render_video failed: {e}")
        raise e

if __name__ == "__main__":
    test_mixed_viz()
