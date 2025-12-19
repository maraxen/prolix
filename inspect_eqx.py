
import json

from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="maraxen/eqx-ff", filename="protein19SB.eqx", repo_type="dataset")
print(f"File path: {path}")

with open(path, "rb") as f:
    header = f.readline().decode()
    print("Header:")
    print(header)
    try:
        data = json.loads(header)
        print("Keys:", data.keys())
        if "num_cmap_maps" in data:
            print("num_cmap_maps:", data["num_cmap_maps"])
        else:
            print("num_cmap_maps NOT FOUND")

        if "cmap_torsions" in data:
            print("cmap_torsions count:", len(data["cmap_torsions"]))
            max_idx = -1
            for t in data["cmap_torsions"]:
                if "map_index" in t:
                    max_idx = max(max_idx, t["map_index"])
            print("Max map_index:", max_idx)
    except Exception as e:
        print("Error parsing JSON:", e)
