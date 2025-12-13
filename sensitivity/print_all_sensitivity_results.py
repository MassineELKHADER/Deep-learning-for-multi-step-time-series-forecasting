import pickle

# ---------- helpers ----------
def print_results(results, sweep_key, name_map=None):
    print(f"\n================ {sweep_key.upper()} RESULTS ================\n")

    if len(results) == 0:
        print("No results found.")
        return

    metric_keys = [k for k in results[0].keys() if k != sweep_key]

    for r in results:
        val = r[sweep_key]
        label = name_map[val] if name_map is not None else val

        print(f"{sweep_key} = {label}")
        for k in metric_keys:
            v = r[k]
            if isinstance(v, float):
                print(f"  {k:15s}: {v:.6f}")
            else:
                print(f"  {k:15s}: {v}")
        print("-" * 40)


# ---------- GAMMA ----------
with open("sensitivity/gamma/results_gamma.pkl", "rb") as f:
    gamma_results = pickle.load(f)

print_results(gamma_results, sweep_key="gamma")


# ---------- ALPHA ----------
with open("sensitivity/alpha/results_alpha.pkl", "rb") as f:
    alpha_results = pickle.load(f)

print_results(alpha_results, sweep_key="alpha")


# ---------- OMEGA ----------
Omega_choices = ["l2", "l1", "asymmetric", "huber"]
Omega_id_to_name = {i: name for i, name in enumerate(Omega_choices)}

with open("sensitivity/Omega/results_Omega_resr.pkl", "rb") as f:
    omega_results = pickle.load(f)

print_results(
    omega_results,
    sweep_key="Omega_id",
    name_map=Omega_id_to_name
)
