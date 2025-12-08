"""Temperature utilities for Parallel Tempering."""
import jax.numpy as jnp

def generate_temperature_ladder(
    n_replicas: int, 
    min_temp: float, 
    max_temp: float,
    geometric: bool = True
) -> jnp.ndarray:
    """Generate a temperature ladder for replica exchange.
    
    Args:
        n_replicas: Number of replicas.
        min_temp: Minimum temperature in Kelvin.
        max_temp: Maximum temperature in Kelvin.
        geometric: If True, use geometric progression (usually optimal for constant heat capacity).
                   If False, use linear progression.
                   
    Returns:
        Array of temperatures [T_0, T_1, ..., T_{N-1}].
    """
    if n_replicas < 1:
        raise ValueError("n_replicas must be >= 1")
    
    if n_replicas == 1:
        return jnp.array([min_temp])

    if geometric:
        # T_i = min * (max/min)^(i/(N-1))
        # log(T_i) = log(min) + i/(N-1) * log(max/min)
        log_min = jnp.log(min_temp)
        log_max = jnp.log(max_temp)
        log_temps = jnp.linspace(log_min, log_max, n_replicas)
        return jnp.exp(log_temps)
    else:
        return jnp.linspace(min_temp, max_temp, n_replicas)
