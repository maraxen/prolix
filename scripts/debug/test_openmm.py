from openmm import app

print("Loading ForceField...")
try:
    omm_ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    print("ForceField loaded.")
except Exception as e:
    print(f"Error loading ForceField: {e}")

print("Done.")
