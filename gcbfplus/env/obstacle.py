import jax
import jax.numpy as jnp

from typing import NamedTuple, Protocol
from jax.scipy.spatial.transform import Rotation
from ..utils.typing import Pos2d, Pos3d, Pos
from ..utils.typing import Array, ObsType, ObsWidth, ObsHeight, ObsTheta, Radius, ObsLength, ObsQuaternion, BoolScalar

RECTANGLE = jnp.zeros(1)
CUBOID = jnp.ones(1)
SPHERE = jnp.ones(1) * 2
CIRCLE = jnp.ones(1) * 3

# Shape constant for MixedObstacle
SHAPE_RECT = 0
SHAPE_CIRCLE = 1


def _rectangle_inside(center: Pos2d, width: ObsWidth, height: ObsHeight, theta: ObsTheta, point: Pos2d, r: Radius = 0.) -> BoolScalar:
    rel_x = point[0] - center[0]
    rel_y = point[1] - center[1]
    rel_xx = jnp.abs(rel_x * jnp.cos(theta) + rel_y * jnp.sin(theta)) - width / 2
    rel_yy = jnp.abs(rel_x * jnp.sin(theta) - rel_y * jnp.cos(theta)) - height / 2
    is_in_down = jnp.logical_and(rel_xx < r, rel_yy < 0)
    is_in_up = jnp.logical_and(rel_xx < 0, rel_yy < r)
    is_out_corner = jnp.logical_and(rel_xx > 0, rel_yy > 0)
    is_in_circle = jnp.sqrt(rel_xx ** 2 + rel_yy ** 2) < r
    is_in = jnp.logical_or(jnp.logical_or(is_in_down, is_in_up), jnp.logical_and(is_out_corner, is_in_circle))
    return is_in


def _rectangle_raytracing(start: Pos2d, end: Pos2d, points: Array) -> Array:
    # beam
    x1 = start[0]
    y1 = start[1]
    x2 = end[0]
    y2 = end[1]

    # edges
    x3 = points[:, 0]
    y3 = points[:, 1]
    x4 = points[[-1, 0, 1, 2], 0]
    y4 = points[[-1, 0, 1, 2], 1]

    det = (x1 - x2) * (y4 - y3) - (y1 - y2) * (x4 - x3)
    # clip det for numerical issues
    det = jnp.sign(det) * jnp.clip(jnp.abs(det), 1e-7, 1e7)
    alphas = ((y4 - y3) * (x1 - x3) - (x4 - x3) * (y1 - y3)) / det
    betas = (-(y1 - y2) * (x1 - x3) + (x1 - x2) * (y1 - y3)) / det
    valids = jnp.logical_and(jnp.logical_and(alphas <= 1, alphas >= 0), jnp.logical_and(betas <= 1, betas >= 0))
    alphas = valids * alphas + (1 - valids) * 1e6
    alphas = jnp.min(alphas)  # reduce the polygon edges dimension
    return alphas


def _circle_inside(center: Pos2d, radius: Radius, point: Pos2d, r: Radius = 0.) -> BoolScalar:
    return jnp.linalg.norm(point - center) <= radius + r


def _circle_raytracing(start: Pos2d, end: Pos2d, center: Pos2d, radius: Radius) -> Array:
    D = end - start
    F = start - center

    A = jnp.dot(D, D)
    B = 2 * jnp.dot(F, D)
    C = jnp.dot(F, F) - radius ** 2

    delta = B ** 2 - 4 * A * C

    valid = (delta >= 0) & (A > 1e-6)

    sqrt_delta = jnp.sqrt(jnp.maximum(delta, 0))

    alpha1 = (-B - sqrt_delta) / (2 * A)
    alpha2 = (-B + sqrt_delta) / (2 * A)

    v1 = (alpha1 >= 0) & (alpha1 <= 1)
    v2 = (alpha2 >= 0) & (alpha2 <= 1)

    a1 = jnp.where(valid & v1, alpha1, 1e6)
    a2 = jnp.where(valid & v2, alpha2, 1e6)

    alpha = jnp.minimum(a1, a2)

    return alpha


class Obstacle(Protocol):
    type: ObsType
    center: Pos
    velocity: Array

    def inside(self, point: Pos, r: Radius = 0.) -> BoolScalar:
        pass

    def raytracing(self, start: Pos, end: Pos) -> Array:
        pass

    def step(self, dt: float) -> "Obstacle":
        pass


class Rectangle(NamedTuple):
    type: ObsType
    center: Pos2d
    width: ObsWidth
    height: ObsHeight
    theta: ObsTheta
    points: Array
    velocity: Pos2d

    @staticmethod
    def create(center: Pos2d, width: ObsWidth, height: ObsHeight, theta: ObsTheta, velocity: Pos2d = None) -> "Rectangle":
        bbox = jnp.array([
            [width / 2, height / 2],
            [-width / 2, height / 2],
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
        ]).T  # (2, 4)

        rot = jnp.array([
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta), jnp.cos(theta)]
        ])

        trans = center[:, None]
        points = jnp.dot(rot, bbox) + trans
        points = points.T

        if velocity is None:
            velocity = jnp.zeros(2)

        return Rectangle(RECTANGLE, center, width, height, theta, points, velocity)

    def step(self, dt: float) -> "Rectangle":
        new_center = self.center + self.velocity * dt
        # Expand velocity to match points shape (..., 4, 2)
        # self.velocity is (..., 2), we need (..., 1, 2)
        v_expanded = jnp.expand_dims(self.velocity, -2)
        new_points = self.points + v_expanded * dt
        return Rectangle(self.type, new_center, self.width, self.height, self.theta, new_points, self.velocity)

    def inside(self, point: Pos2d, r: Radius = 0.) -> BoolScalar:
        return _rectangle_inside(self.center, self.width, self.height, self.theta, point, r)

    def raytracing(self, start: Pos2d, end: Pos2d) -> Array:
        return _rectangle_raytracing(start, end, self.points)


class Cuboid(NamedTuple):
    type: ObsType
    center: Pos3d
    length: ObsLength
    width: ObsWidth
    height: ObsHeight
    rotation: Rotation
    points: Array
    velocity: Pos3d

    @staticmethod
    def create(
            center: Pos3d, length: ObsLength, width: ObsWidth, height: ObsHeight, quaternion: ObsQuaternion,
            velocity: Pos3d = None
    ) -> "Cuboid":
        bbox = jnp.array([
            [-length / 2, -width / 2, -height / 2],
            [length / 2, -width / 2, -height / 2],
            [length / 2, width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
            [-length / 2, -width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [length / 2, width / 2, height / 2],
            [-length / 2, width / 2, height / 2],
        ])  # (8, 3)

        rotation = Rotation.from_quat(quaternion)
        points = rotation.apply(bbox) + center
        if velocity is None:
            velocity = jnp.zeros(3)
        return Cuboid(CUBOID, center, length, width, height, rotation, points, velocity)

    def step(self, dt: float) -> "Cuboid":
        new_center = self.center + self.velocity * dt
        # Expand velocity to match points shape (..., 8, 3)
        # self.velocity is (..., 3), we need (..., 1, 3)
        v_expanded = jnp.expand_dims(self.velocity, -2)
        new_points = self.points + v_expanded * dt
        return Cuboid(self.type, new_center, self.length, self.width, self.height, self.rotation, new_points, self.velocity)

    def inside(self, point: Pos3d, r: Radius = 0.) -> BoolScalar:
        # transform the point to the cuboid frame
        rot = self.rotation.as_matrix()
        rot_inv = jnp.linalg.inv(rot)
        point = jnp.dot(rot_inv, point - self.center)

        # check if the point is inside the cuboid
        is_in_height = ((-self.length / 2 < point[0]) & (point[0] < self.length / 2)) & \
                       ((-self.width / 2 < point[1]) & (point[1] < self.width / 2)) & \
                       ((-self.height / 2 - r < point[2]) & (point[2] < self.height / 2 + r))
        is_in_length = ((-self.length / 2 - r < point[0]) & (point[0] < self.length / 2 + r)) & \
                       ((-self.width / 2 < point[1]) & (point[1] < self.width / 2)) & \
                       ((-self.height / 2 < point[2]) & (point[2] < self.height / 2))
        is_in_width = ((-self.length / 2 < point[0]) & (point[0] < self.length / 2)) & \
                      ((-self.width / 2 - r < point[1]) & (point[1] < self.width / 2 + r)) & \
                      ((-self.height / 2 < point[2]) & (point[2] < self.height / 2))
        is_in = is_in_height | is_in_length | is_in_width

        # check if the sphere intersects with the edges
        edge_order = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0],
                                [4, 5], [5, 6], [6, 7], [7, 4],
                                [0, 4], [1, 5], [2, 6], [3, 7]])
        edges = self.points[edge_order]

        def intersect_edge(edge: Array) -> BoolScalar:
            assert edge.shape == (2, 3)
            dot_prod = jnp.dot(edge[1] - edge[0], point - edge[0])
            frac = dot_prod / ((jnp.linalg.norm(edge[1] - edge[0])) ** 2)
            frac = jnp.clip(frac, 0, 1)
            closest_point = edge[0] + frac * (edge[1] - edge[0])
            dist = jnp.linalg.norm(closest_point - point)
            return dist <= r

        is_intersect = jnp.any(jax.vmap(intersect_edge)(edges))
        return is_in | is_intersect

    def raytracing(self, start: Pos3d, end: Pos3d) -> Array:
        # beams
        x1, y1, z1 = start[0], start[1], start[2]
        x2, y2, z2 = end[0], end[1], end[2]

        # those are for edges
        # point order for the base 0~7 (first 0-x-xy-y for the lower level, then 0-x-xy-y for the upper level)
        # face order: bottom, left, right, upper, outer left, outer right
        x3 = self.points[[0, 0, 0, 6, 6, 6], 0]
        y3 = self.points[[0, 0, 0, 6, 6, 6], 1]
        z3 = self.points[[0, 0, 0, 6, 6, 6], 2]

        x4 = self.points[[1, 1, 3, 5, 5, 7], 0]
        y4 = self.points[[1, 1, 3, 5, 5, 7], 1]
        z4 = self.points[[1, 1, 3, 5, 5, 7], 2]

        x5 = self.points[[3, 4, 4, 7, 2, 2], 0]
        y5 = self.points[[3, 4, 4, 7, 2, 2], 1]
        z5 = self.points[[3, 4, 4, 7, 2, 2], 2]

        '''
        # solve the equation
        # x = x1 + alpha * (x2 - x1) = x3 + beta * (x4 - x3) + gamma * (x5 - x3)
        # y = y1 + alpha * (y2 - y1) = y3 + beta * (y4 - y3) + gamma * (y5 - y3) 
        # z = z1 + alpha * (z2 - z1) = z3 + beta * (z4 - z3) + gamma * (z5 - z3)
        # equivalent to solve
        # [x1 - x2  x4 - x3  x5 - x3]  alpha  =  [x1 - x3] 
        # [y1 - y2  y4 - y3  y5 - y3]  beta      [y1 - y3]
        # [z1 - z2  z4 - z3  z5 - z3]  gamma     [z1 - z3]
        # solve by (alpha beta gamma)^T = A^{-1} b

        # A^{-1} = 1/det * [(y4-y3)*(z5-z3)-(y5-y3)*(z4-z3)      -[(x4-x3)*(z5-z3)-(z4-z3)*(x5-x3)]      (x4-x3)*(y5-y3)-(y4-y3)*(x5-x3)]
        #                  [-[(y1-y2)*(z5-z3)-(z1-z2)*(y5-y3)]   (x1-x2)*(z5-z3)-(z1-z2)*(x5-x3)      -[(x1-x2)*(y5-y3)-(y1-y2)*(x5-x3)]]
        #                  [(y1-y2)*(z4-z3)-(y4-y3)*(z1-z2)      -[(x1-x2)*(z4-z3)-(z1-z2)*(x4-x3)]      (x1-x2)*(y4-y3)-(y1-y2)*(x4-x3)]
        '''

        det = (x1 - x2) * (y4 - y3) * (z5 - z3) + (x4 - x3) * (y5 - y3) * (z1 - z2) + (y1 - y2) * (z4 - z3) * (
                x5 - x3) - (y1 - y2) * (x4 - x3) * (z5 - z3) - (z4 - z3) * (y5 - y3) * (x1 - x2) - (x5 - x3) * (
                      y4 - y3) * (z1 - z2)
        # clip det for numerical issues
        det = jnp.sign(det) * jnp.clip(jnp.abs(det), 1e-7, 1e7)
        adj_00 = (y4 - y3) * (z5 - z3) - (y5 - y3) * (z4 - z3)
        adj_01 = -((x4 - x3) * (z5 - z3) - (z4 - z3) * (x5 - x3))
        adj_02 = (x4 - x3) * (y5 - y3) - (y4 - y3) * (x5 - x3)
        adj_10 = -((y1 - y2) * (z5 - z3) - (z1 - z2) * (y5 - y3))
        adj_11 = (x1 - x2) * (z5 - z3) - (z1 - z2) * (x5 - x3)
        adj_12 = -((x1 - x2) * (y5 - y3) - (y1 - y2) * (x5 - x3))
        adj_20 = (y1 - y2) * (z4 - z3) - (y4 - y3) * (z1 - z2)
        adj_21 = -((x1 - x2) * (z4 - z3) - (z1 - z2) * (x4 - x3))
        adj_22 = (x1 - x2) * (y4 - y3) - (y1 - y2) * (x4 - x3)
        alphas = 1 / det * (adj_00 * (x1 - x3) + adj_01 * (y1 - y3) + adj_02 * (z1 - z3))
        betas = 1 / det * (adj_10 * (x1 - x3) + adj_11 * (y1 - y3) + adj_12 * (z1 - z3))
        gammas = 1 / det * (adj_20 * (x1 - x3) + adj_21 * (y1 - y3) + adj_22 * (z1 - z3))
        valids = jnp.logical_and(
            jnp.logical_and(jnp.logical_and(alphas <= 1, alphas >= 0), jnp.logical_and(betas <= 1, betas >= 0)),
            jnp.logical_and(gammas <= 1, gammas >= 0)
        )
        alphas = valids * alphas + (1 - valids) * 1e6
        alphas = jnp.min(alphas)  # reduce the polygon edges dimension
        return alphas


class Sphere(NamedTuple):
    type: ObsType
    center: Pos3d
    radius: Radius
    velocity: Pos3d

    @staticmethod
    def create(center: Pos3d, radius: Radius, velocity: Pos3d = None) -> "Sphere":
        if velocity is None:
            velocity = jnp.zeros(3)
        return Sphere(SPHERE, center, radius, velocity)

    def step(self, dt: float) -> "Sphere":
        new_center = self.center + self.velocity * dt
        return Sphere(self.type, new_center, self.radius, self.velocity)

    def inside(self, point: Pos3d, r: Radius = 0.) -> BoolScalar:
        return jnp.linalg.norm(point - self.center) <= self.radius + r

    def raytracing(self, start: Pos3d, end: Pos3d) -> Array:
        x1, y1, z1 = start[0], start[1], start[2]
        x2, y2, z2 = end[0], end[1], end[2]
        xc, yc, zc = self.center[0], self.center[1], self.center[2]
        r = self.radius

        '''
        # solve the equation
        # x = x1 + alpha * (x2 - x1) = xc + r * sin(gamma) * cos(theta)
        # y = y1 + alpha * (y2 - y1) = yc + r * sin(gamma) * sin(theta)
        # z = z1 + alpha * (z2 - z1) = zc + r * cos(gamma)
        # equivalent to solve (eliminate theta using sin^2(sin^2+cos^2) +cos^2 ...=1)
        # [(x2-x1)^2+(y2-y1)^2+(z2-z1)^2]alpha^2+2[(x2-x1)(x1-xc)+(y2-y1)(y1-yc)+(z2-z1)(z1-zc)]alpha+(x1-xc)^2+(y1-yc)^2+(z1-zc)^2-r^2=0
        # A alpha^2 + B alpha + C = 0
        # check delta = B^2-4AC
        # alpha = ...
        # take valid min
        '''
        lidar_rmax = jnp.linalg.norm(end - start)
        A = lidar_rmax ** 2  # (x2-x1)^2+(y2-y1)^2
        B = 2 * ((x2 - x1) * (x1 - xc) + (y2 - y1) * (y1 - yc) + (z2 - z1) * (z1 - zc))
        C = (x1 - xc) ** 2 + (y1 - yc) ** 2 + (z1 - zc) ** 2 - r ** 2

        delta = B ** 2 - 4 * A * C
        valid1 = delta >= 0

        alpha1 = (-B - jnp.sqrt(delta * valid1)) / (2 * A) * valid1 + (1 - valid1)
        alpha2 = (-B + jnp.sqrt(delta * valid1)) / (2 * A) * valid1 + (1 - valid1)
        alpha1_tilde = (alpha1 >= 0) * alpha1 + (alpha1 < 0) * 1
        alpha2_tilde = (alpha2 >= 0) * alpha2 + (alpha2 < 0) * 1
        alphas = jnp.minimum(alpha1_tilde, alpha2_tilde)
        alphas = jnp.clip(alphas, 0, 1)
        alphas = valid1 * alphas + (1 - valid1) * 1e6
        return alphas


class Circle(NamedTuple):
    type: ObsType
    center: Pos2d
    radius: Radius
    velocity: Pos2d

    @staticmethod
    def create(center: Pos2d, radius: Radius, velocity: Pos2d = None) -> "Circle":
        if velocity is None:
            velocity = jnp.zeros(2)
        return Circle(CIRCLE, center, radius, velocity)

    def step(self, dt: float) -> "Circle":
        new_center = self.center + self.velocity * dt
        return Circle(self.type, new_center, self.radius, self.velocity)

    def inside(self, point: Pos2d, r: Radius = 0.) -> BoolScalar:
        return _circle_inside(self.center, self.radius, point, r)

    def raytracing(self, start: Pos2d, end: Pos2d) -> Array:
        return _circle_raytracing(start, end, self.center, self.radius)


class MixedObstacle(NamedTuple):
    type: ObsType
    center: Pos2d
    velocity: Pos2d
    # Shape parameters
    shape_type: Array  # 0: Rectangle, 1: Circle
    width: ObsWidth     # For Rectangle
    height: ObsHeight   # For Rectangle
    theta: ObsTheta     # For Rectangle
    radius: Radius      # For Circle
    points: Array       # For Rectangle (vertices)
    inspection_target: Array # Boolean flag: True if this obstacle is an inspection target

    @staticmethod
    def create(
        center: Pos2d, 
        velocity: Pos2d, 
        shape_type: int,
        width: float = 0.0, 
        height: float = 0.0, 
        theta: float = 0.0, 
        radius: float = 0.0,
        inspection_target: bool = False
    ) -> "MixedObstacle":
        
        # Calculate points for Rectangle (always calculate, ignore if Circle)
        bbox = jnp.array([
            [width / 2, height / 2],
            [-width / 2, height / 2],
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
        ]).T  # (2, 4)

        rot = jnp.array([
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta), jnp.cos(theta)]
        ])

        trans = center[:, None]
        points = jnp.dot(rot, bbox) + trans
        points = points.T
        
        # If velocity is None, calling code should handle it, but here we expect it to be passed.
        # However, for safety:
        if velocity is None:
            velocity = jnp.zeros(2)

        # type field is generic ObsType, we can use RECTANGLE or CIRCLE, but since it's mixed...
        # Let's just use RECTANGLE as placeholder or a new constant. 
        # But `type` in Protocol is mostly for identification if needed.
        # We'll use the shape_type field for logic.
        
        return MixedObstacle(
            RECTANGLE, # Placeholder
            center, 
            velocity, 
            jnp.array(shape_type, dtype=jnp.int32), 
            width, 
            height,
            theta, 
            radius, 
            points,
            jnp.array(inspection_target, dtype=bool)
        )

    def step(self, dt: float) -> "MixedObstacle":
        new_center = self.center + self.velocity * dt
        
        # Update points for Rectangle
        # self.velocity is (..., 2), we need (..., 1, 2)
        v_expanded = jnp.expand_dims(self.velocity, -2)
        new_points = self.points + v_expanded * dt
        
        return MixedObstacle(
            self.type, 
            new_center, 
            self.velocity, 
            self.shape_type, 
            self.width, 
            self.height, 
            self.theta, 
            self.radius, 
            self.radius, 
            new_points,
            self.inspection_target
        )

    def inside(self, point: Pos2d, r: Radius = 0.) -> BoolScalar:
        # Branch based on shape_type
        # Assuming vmapped, so shape_type is a scalar for this instance
        
        def inside_rect(p):
            return _rectangle_inside(self.center, self.width, self.height, self.theta, p, r)
            
        def inside_circle(p):
            return _circle_inside(self.center, self.radius, p, r)
            
        return jax.lax.cond(
            self.shape_type == SHAPE_CIRCLE,
            inside_circle,
            inside_rect,
            point
        )

    def raytracing(self, start: Pos2d, end: Pos2d) -> Array:
        def trace_rect(args):
            s, e = args
            return _rectangle_raytracing(s, e, self.points)
            
        def trace_circle(args):
            s, e = args
            return _circle_raytracing(s, e, self.center, self.radius)
            
        return jax.lax.cond(
            self.shape_type == SHAPE_CIRCLE,
            trace_circle,
            trace_rect,
            (start, end)
        )
