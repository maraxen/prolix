import sys
import pytest

def test_circular_imports():
    # Attempt to import prolix.typing first
    import prolix.typing
    assert 'prolix.typing' in sys.modules

    # Import core physics modules
    # These imports should not raise ImportError or AttributeError
    try:
        import prolix.physics.system
        import prolix.physics.simulate
        import prolix.physics.neighbor_list
        import prolix.physics.integrator_builder
        import prolix.physics.bonded
        import prolix.physics.electrostatic_methods
        import prolix.physics.generalized_born
        import prolix.physics.cmap
    except (ImportError, AttributeError) as e:
        pytest.fail(f"Circular import detected: {e}")

    # Verify modules were loaded
    assert 'prolix.physics.system' in sys.modules
    assert 'prolix.physics.simulate' in sys.modules
    assert 'prolix.physics.neighbor_list' in sys.modules
    assert 'prolix.physics.integrator_builder' in sys.modules
    assert 'prolix.physics.bonded' in sys.modules
    assert 'prolix.physics.electrostatic_methods' in sys.modules
    assert 'prolix.physics.generalized_born' in sys.modules
    assert 'prolix.physics.cmap' in sys.modules

    print("Circular import test passed!")

if __name__ == "__main__":
    test_circular_imports()
