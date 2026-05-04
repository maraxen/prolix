import jax
import jax.numpy as jnp
from prolix.physics import pme

def test_pme_differentiability():
    # Setup a small system
    n_atoms = 10
    positions = jax.random.uniform(jax.random.PRNGKey(0), (n_atoms, 3)) * 10.0
    charges = jax.random.normal(jax.random.PRNGKey(1), (n_atoms,))
    atom_mask = jnp.ones(n_atoms, dtype=bool)
    box_size = jnp.array([10.0, 10.0, 10.0])
    alpha = 0.34
    grid_dims = (16, 16, 16)
    order = 4

    def energy_fn(pos, q, a):
        # We need to call spme_energy_with_forces directly or through a wrapper
        return pme.spme_energy_with_forces(pos, q, atom_mask, box_size, grid_dims, a, order)

    # 1. Gradient w.r.t positions
    grad_pos_fn = jax.grad(energy_fn, argnums=0)
    g_pos = grad_pos_fn(positions, charges, alpha)
    print(f"Grad positions finite: {jnp.all(jnp.isfinite(g_pos))}")
    print(f"Grad positions sum: {jnp.sum(jnp.abs(g_pos))}")

    # 2. Gradient w.r.t charges
    grad_q_fn = jax.grad(energy_fn, argnums=1)
    g_q = grad_q_fn(positions, charges, alpha)
    print(f"Grad charges finite: {jnp.all(jnp.isfinite(g_q))}")
    print(f"Grad charges sum: {jnp.sum(jnp.abs(g_q))}")

    # 3. Gradient w.r.t alpha
    try:
        grad_alpha_fn = jax.grad(energy_fn, argnums=2)
        g_alpha = grad_alpha_fn(positions, charges, alpha)
        print(f"Grad alpha finite: {jnp.all(jnp.isfinite(g_alpha))}")
        print(f"Grad alpha: {g_alpha}")
    except Exception as e:
        print(f"Grad alpha failed: {e}")

    # 3b. Gradient w.r.t box_size
    def energy_box_fn(box):
        return pme.spme_energy_with_forces(positions, charges, atom_mask, box, grid_dims, alpha, order)
    
    try:
        grad_box_fn = jax.grad(energy_box_fn)
        g_box = grad_box_fn(box_size)
        print(f"Grad box finite: {jnp.all(jnp.isfinite(g_box))}")
        print(f"Grad box: {g_box}")
    except Exception as e:
        print(f"Grad box failed: {e}")

    # 4. Hessian-vector product (Positions)
    v = jax.random.normal(jax.random.PRNGKey(2), positions.shape)
    # Force sensitivity to positions (H_RR * v)
    hvp_pos_fn = jax.grad(lambda r: jnp.sum(grad_pos_fn(r, charges, alpha) * v))
    try:
        h = hvp_pos_fn(positions)
        print(f"HvP positions finite: {jnp.all(jnp.isfinite(h))}")
    except Exception as e:
        print(f"HvP positions failed: {e}")

    # 5. Force sensitivity to charges (d^2E/dr dq * v)
    # We want d/dq ( dE/dr * v )
    force_sensitivity_q_fn = jax.grad(lambda q: jnp.sum(grad_pos_fn(positions, q, alpha) * v))
    try:
        s_q = force_sensitivity_q_fn(charges)
        print(f"Force sensitivity to charges finite: {jnp.all(jnp.isfinite(s_q))}")
        print(f"Force sensitivity to charges sum: {jnp.sum(jnp.abs(s_q))}")
    except Exception as e:
        print(f"Force sensitivity to charges failed: {e}")

    # 6. Force sensitivity to alpha (d^2E/dr dalpha * v)
    force_sensitivity_a_fn = jax.grad(lambda a: jnp.sum(grad_pos_fn(positions, charges, a) * v))
    try:
        s_a = force_sensitivity_a_fn(alpha)
        print(f"Force sensitivity to alpha finite: {jnp.all(jnp.isfinite(s_a))}")
    except Exception as e:
        print(f"Force sensitivity to alpha failed: {e}")

if __name__ == "__main__":
    test_pme_differentiability()
