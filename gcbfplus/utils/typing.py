# from jaxtyping import Array, Bool, Float, Int, Shaped
from typing import Dict, TypeVar, Any, List, Union
from jax import Array
import jax.numpy as jnp
from numpy import ndarray

# Dummy aliases for jaxtyping
Bool = Any
Float = Any
Int = Any
Shaped = Any


# jax types
PRNGKey = Any

BoolScalar = Any
ABool = Any

# environment types
Action = Any
Reward = Any
Cost = Any
Done = BoolScalar
Info = Dict[str, Any]
EdgeIndex = Any
AgentState = Any
State = Any
Node = Any
EdgeAttr = Any
Pos2d = Union[Array, ndarray]
Pos3d = Union[Array, ndarray]
Pos = Union[Pos2d, Pos3d]
Radius = Union[Array, float]


# neural network types
Params = TypeVar("Params", bound=core.FrozenDict[str, Any])

# obstacles
ObsType = Any
ObsWidth = Any
ObsHeight = Any
ObsLength = Any
ObsTheta = Any
ObsQuaternion = Any
