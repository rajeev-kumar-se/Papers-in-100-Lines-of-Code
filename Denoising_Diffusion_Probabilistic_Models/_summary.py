import json

import numpy as np

m = json.load(open("eval/metrics.json"))
g = m["classifier"]["generated"]["hist"]
r = m["classifier"]["real"]["hist"]
print("class hist gen ", [round(x, 3) for x in g])
print("class hist real", [round(x, 3) for x in r])
for k in ("generated", "real"):
    d = np.array(m["nn_distance"][k])
    print(f"nn {k}: median {np.median(d):.2f} p05 {np.percentile(d, 5):.2f} p95 {np.percentile(d, 95):.2f} min {d.min():.2f}")
print("moment:", [(x["t"], round(x["mean_max_abs_err"], 4), round(x["std_max_abs_err"], 4)) for x in m["moment_check"]])
pt = m["per_timestep"]
print("per_t first/last", (pt["t"][0], round(pt["mse"][0], 4)), (pt["t"][-1], round(pt["mse"][-1], 4)))
print("crosscheck", m["fid_crosscheck"])
print("sqrt", m["fid_sqrt_method"])
print("seeds", m["fid_stats"])
print("PR", m["precision_recall"])
print("wall_min", round(m["wall_clock_s"] / 60, 2))
print("fid", {k: round(v, 3) for k, v in m["fid"].items()})
print("kid", {k: round(v, 5) for k, v in m["kid"].items()})
print("fid_vs_n", m["fid_vs_n"])
