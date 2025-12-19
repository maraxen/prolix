"""Batch Validation for Prolix vs OpenMM Physics.
"""

import pandas as pd
import verify_end_to_end_physics
from termcolor import colored

# Proteins to test (<100 AA)
PROTEINS = [
    "1UAO", # 12 AA (Hydride, done)
    # "1L2Y", # Trp-Cage (20 AA)
    "1CRN", # Crambin (46 AA)
    "1VII", # Villin Headpiece (36 AA)
    "2JOF", # Pin1 WW (34 AA)
    # "1UBQ"  # Ubiquitin (76 AA)
]

def run_batch():
    results = []

    print(colored("Starting Batch Validation...", "cyan"))

    for pdb in PROTEINS:
        print(colored(f"\nProcessing {pdb}...", "yellow"))
        try:
            res = verify_end_to_end_physics.run_verification(pdb_code=pdb, return_results=True)
            if res:
                results.append(res)
            else:
                print(colored(f"Failed to run {pdb}", "red"))
        except Exception as e:
            print(colored(f"Exception running {pdb}: {e}", "red"))
            import traceback
            traceback.print_exc()

    df = pd.DataFrame(results)
    print("\nBatch Results:")
    print(df)

    # Save
    df.to_csv("batch_physics_results.csv", index=False)

    # Stats
    if not df.empty:
        mae = (df["omm_energy"] - df["jax_energy"]).abs().mean()
        corr = df["omm_energy"].corr(df["jax_energy"])
        print(f"\nMAE Total Energy: {mae:.4f} kcal/mol")
        print(f"Correlation (R): {corr:.4f}")

    print(colored("\nDone.", "green"))

if __name__ == "__main__":
    run_batch()
