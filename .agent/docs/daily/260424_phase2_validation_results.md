An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
/home/marielle/projects/prolix/src/prolix/physics/cmap.py:39: UserWarning: Explicitly requested dtype float64 requested in array is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  WT = jnp.array(

v0.3.0 PHASE 2: CONSTRAINT-AWARE LANGEVIN VALIDATION


[TEST 1/3] Baseline Control (dt=0.5fs)

======================================================================
Validation: dt=0.5fs, duration=50ps, 4 waters
======================================================================
Timesteps: 100000 (dt=0.010227 AKMA)

Running 100000 steps...
Traceback (most recent call last):
  File "/home/marielle/projects/prolix/scripts/validate_constraint_aware_langevin.py", line 319, in <module>
    exit(main())
         ^^^^^^
  File "/home/marielle/projects/prolix/scripts/validate_constraint_aware_langevin.py", line 259, in main
    results_05 = run_validation_test(dt_fs=0.5, duration_ps=50.0, n_waters=4)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/marielle/projects/prolix/scripts/validate_constraint_aware_langevin.py", line 160, in run_validation_test
    state = apply_fn(state)
            ^^^^^^^^^^^^^^^
  File "/home/marielle/projects/prolix/src/prolix/physics/settle.py", line 683, in apply_fn
    position = _langevin_step_a(state.position, momentum, state.mass, _dt, shift_fn)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/marielle/projects/prolix/src/prolix/physics/settle.py", line 904, in _langevin_step_a
    return shift_fn(position, 0.5 * dt * velocity)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: run_validation_test.<locals>.mock_shift_fn() takes 1 positional argument but 2 were given
