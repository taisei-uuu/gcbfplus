import jax
import jax.numpy as jnp
import numpy as np
from gcbfplus.env import DoubleIntegrator
from gcbfplus.env.obstacle import SHAPE_RECT, SHAPE_CIRCLE

def test_mixed_obstacles():
    print("Testing Mixed Obstacles...")
    
    # Define mixed config
    # 1. Circle, static, radius 0.2, at [0.5, 0.5]
    # 2. Rectangle, dynamic (vel=[0.1, 0]), width 0.2, height 0.4, at [-0.5, -0.5]
    
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
                "velocity": [0.1, 0.0],
                "theta": 0.0
            }
        ],
        "agents": {
            "start": [[0.0, 0.0]], # 1 agent
            "goal": [[1.0, 1.0]]
        }
    }
    
    params = DoubleIntegrator.PARAMS.copy()
    params.update({
        "fixed_config": fixed_config,
        "n_obs": 2, # ignored but good to set
        "num_agents": 1
    })
    
    env = DoubleIntegrator(num_agents=1, area_size=2.0, params=params)
    
    key = jax.random.PRNGKey(0)
    graph = env.reset(key)
    
    obstacles = graph.env_states.obstacle
    
    print("Obstacles created:", obstacles)
    
    # Check properties
    # Convert to numpy for easy check
    centers = np.array(obstacles.center)
    velocities = np.array(obstacles.velocity)
    shapes = np.array(obstacles.shape_type)
    radii = np.array(obstacles.radius)
    widths = np.array(obstacles.width)
    
    # Obs 1: Circle
    assert np.allclose(centers[0], [0.5, 0.5])
    assert np.allclose(velocities[0], [0.0, 0.0])
    assert shapes[0] == SHAPE_CIRCLE
    assert np.isclose(radii[0], 0.2)
    
    # Obs 2: Rect
    assert np.allclose(centers[1], [-0.5, -0.5])
    assert np.allclose(velocities[1], [0.1, 0.0])
    assert shapes[1] == SHAPE_RECT
    assert np.isclose(widths[1], 0.2)
    # assert np.isclose(obstacles.height[1], 0.4) # height accessed if I implemented it as such, but MixedObstacle has .height field
    
    print("Initial state verified.")
    
    # Step environment
    action = jnp.zeros((1, 2))
    next_graph, _, _, _, _ = env.step(graph, action)
    
    next_obstacles = next_graph.env_states.obstacle
    next_centers = np.array(next_obstacles.center)
    
    # Verify movement
    # Obs 1 (static) should be same
    assert np.allclose(next_centers[0], [0.5, 0.5])
    
    # Obs 2 (dynamic) should move: vel * dt
    dt = env.dt
    expected_pos = np.array([-0.5, -0.5]) + np.array([0.1, 0.0]) * dt
    assert np.allclose(next_centers[1], expected_pos)
    
    print(f"Step verified. Obs 2 moved from [-0.5, -0.5] to {next_centers[1]}")
    print("Test Passed!")

if __name__ == "__main__":
    test_mixed_obstacles()
