from typing import Optional

import jax.numpy as jnp

from .base import MultiAgentEnv
from .single_integrator import SingleIntegrator
from .double_integrator import DoubleIntegrator
from .linear_drone import LinearDrone
from .dubins_car import DubinsCar
from .crazyflie import CrazyFlie


ENV = {
    'SingleIntegrator': SingleIntegrator,
    'DoubleIntegrator': DoubleIntegrator,
    'LinearDrone': LinearDrone,
    'DubinsCar': DubinsCar,
    'CrazyFlie': CrazyFlie,
}


DEFAULT_MAX_STEP = 256


def make_env(
        env_id: str,
        num_agents: int,
        area_size: float = None,
        max_step: int = None,
        max_travel: Optional[float] = None,
        num_obs: Optional[int] = None,
        n_rays: Optional[int] = None,
        formation_mode: bool = False,  # 追加
        formation_offsets: Optional[list] = None,  # 追加
) -> MultiAgentEnv:
    assert env_id in ENV.keys(), f'Environment {env_id} not implemented.'
    params = ENV[env_id].PARAMS.copy()
    max_step = DEFAULT_MAX_STEP if max_step is None else max_step
    if num_obs is not None:
        params['n_obs'] = num_obs
    if n_rays is not None:
        params['n_rays'] = n_rays
    
    # フォーメーションパラメータを追加
    if formation_mode:
        params["formation_mode"] = True
        if formation_offsets is None:
            # デフォルトのオフセット（三角形フォーメーション）
            # エージェント1: 右側、エージェント2: 左側
            formation_offsets = [[0.3, 0.0], [-0.3, 0.0]]
        # JAX配列に変換して保存
        params["formation_offsets"] = jnp.array(formation_offsets)
    else:
        params["formation_mode"] = False
        params["formation_offsets"] = None
    return ENV[env_id](
        num_agents=num_agents,
        area_size=area_size,
        max_step=max_step,
        max_travel=max_travel,
        dt=0.03,
        params=params
    )
