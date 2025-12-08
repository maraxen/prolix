
try:
    import py3Dmol
    view = py3Dmol.view()
    print("Methods:", dir(view))
except ImportError:
    print("py3Dmol not installed")
