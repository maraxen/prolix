

try:
    import py2Dmol
    view = py2Dmol.view()
    print("Methods:", dir(view))
except ImportError:
    print("py2Dmol not installed")
