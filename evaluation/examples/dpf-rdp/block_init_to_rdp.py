# Quickly convert to RDP for debugging purposes
# Usage: `python block_init_to_rdp.py add-block.yaml`

import yaml
import numpy as np
import sys

ALPHAS = [1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64, 1e6]


def to_rdp(epsDel):
    r = []
    ε = epsDel["epsDel"]["epsilon"]
    δ = epsDel["epsDel"]["delta"]
    for α in ALPHAS:
        ε_α = max(ε + np.log(δ) / (α - 1), 0)
        r.append(
            {
                "alpha": α,
                "epsilon": float(ε_α),
            }
        )

    return {"renyi": r}


with open(sys.argv[1], "r") as block:
    d = yaml.load(block.read(), Loader=yaml.SafeLoader)
    d["spec"]["initialBudget"] = to_rdp(d["spec"]["initialBudget"])
    with open(sys.argv[1][0 : -len(".yaml")] + "_rdp.yaml", "w") as new_block:
        yaml.dump(d, new_block)