
class Circle(NamedTuple):
    type: ObsType
    center: Pos2d
    radius: Radius

    @staticmethod
    def create(center: Pos2d, radius: Radius) -> "Circle":
        return Circle(CIRCLE, center, radius)

    def inside(self, point: Pos2d, r: Radius = 0.) -> BoolScalar:
        return jnp.linalg.norm(point - self.center) <= self.radius + r

    def raytracing(self, start: Pos2d, end: Pos2d) -> Array:
        # Solving intersection of line segment P = start + alpha * (end - start)
        # with circle |P - center|^2 = radius^2.
        # This is quadratic equation in alpha at most.
        # Let D = end - start, F = start - center
        # |F + alpha * D|^2 = r^2
        # (F + alpha*D).(F + alpha*D) = r^2
        # |D|^2 alpha^2 + 2(F.D) alpha + |F|^2 - r^2 = 0
        # A alpha^2 + B alpha + C = 0
        
        D = end - start
        F = start - self.center
        
        A = jnp.dot(D, D)
        B = 2 * jnp.dot(F, D)
        C = jnp.dot(F, F) - self.radius**2
        
        delta = B**2 - 4 * A * C
        
        # We only care if delta >= 0
        valid = (delta >= 0) & (A > 1e-6)
        
        sqrt_delta = jnp.sqrt(jnp.maximum(delta, 0))
        
        # Two solutions
        alpha1 = (-B - sqrt_delta) / (2 * A)
        alpha2 = (-B + sqrt_delta) / (2 * A)
        
        # Valid intersection must be within [0, 1]
        # We want the smallest positive alpha if multiple exist
        # But ray tracing usually returns distance/fraction to first hit.
        
        # Check validity for alpha1
        v1 = (alpha1 >= 0) & (alpha1 <= 1)
        # Check validity for alpha2
        v2 = (alpha2 >= 0) & (alpha2 <= 1)
        
        # If both valid, take smaller. If only one, take it. If neither, no hit.
        # We can set invalid alphas to infinity (or 1e6)
        
        a1 = jnp.where(valid & v1, alpha1, 1e6)
        a2 = jnp.where(valid & v2, alpha2, 1e6)
        
        alpha = jnp.minimum(a1, a2)
        
        # If alpha is 1e6, it means no hit
        # The caller expects something <= 1 for hit, or 1e6 for no hit.
        return alpha
