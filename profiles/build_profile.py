"""
Build preloaded DMSF profile JSON from lab profiling CSV.

Usage:
  python profiles/build_profile.py --csv lab_profile.csv --edge-col 2 --cloud-col 1 --out server9dev2_s10dev1.json

CSV format expected (exported from Excel):
  layer_index, layer_type, <device_col>, ...
  0, Conv, 0.672873, ...
  ...

Or you can hardcode the values below directly for each device combo.
"""

import json
import argparse
import csv
import sys
from pathlib import Path

# DMSF split points (backbone layer indices in yolo26n)
SPLIT_POINTS = [3, 5, 7, 10]

# Hardcoded lab profiling data (from Excel image)
# Format: device_id -> list of 24 per-layer times (ms, batch=32)
#
# SERVER 9 devices (edge candidates):
LAB_DATA = {
    # SERVER 9 → edge devices (machine-2 .. machine-10)
    "machine-2": [0.632648, 0.613632, 1.665783, 0.497527, 0.805163, 0.370122, 0.407436, 0.153539,
                  0.264231, 0.337384, 0.353267, 0.157564, 0.050527, 0.52476, 0.088686, 0.123981,
                  1.08965, 0.127852, 0.016081, 0.470386, 0.087953, 0.007321, 0.486597, 3.6768057],
    "machine-3": [0.614831, 0.592691, 1.755248, 0.572218, 0.864683, 0.375688, 0.472662, 0.155149,
                  0.267167, 0.414424, 0.359957, 0.16135, 0.051955, 0.528222, 0.106734, 0.129095,
                  1.002707, 0.105793, 0.014804, 0.464446, 0.087472, 0.006927, 0.462401, 3.418063],
    "machine-4": [0.613534, 0.595246, 1.715516, 0.564741, 0.850708, 0.372867, 0.427928, 0.155173,
                  0.263105, 0.418231, 0.357977, 0.229709, 0.051738, 0.534366, 0.105139, 0.132612,
                  1.045779, 0.107135, 0.013949, 0.440991, 0.08677, 0.007503, 0.453071, 3.507187],
    "machine-5": [0.830647, 0.829031, 2.303999, 0.798803, 1.121759, 0.549891, 0.540792, 0.237312,
                  0.376502, 0.622485, 0.463347, 0.273506, 0.067306, 0.718651, 0.15167, 0.177203,
                  1.339995, 0.146156, 0.016914, 0.586882, 0.126512, 0.008653, 0.636149, 4.583206],
    "machine-6": [0.850925, 0.700756, 2.338063, 0.741838, 1.036324, 0.5128, 0.568137, 0.24307,
                  0.382012, 0.629724, 0.47677, 0.196012, 0.066806, 0.693734, 0.135998, 0.177998,
                  1.321085, 0.151106, 0.018072, 0.613787, 0.128, 0.00845, 0.664648, 4.431388],
    "machine-7": [0.826741, 0.814398, 2.28295, 0.794509, 1.108342, 0.560426, 0.561048, 0.227987,
                  0.378584, 0.641373, 0.526807, 0.257863, 0.063483, 0.706496, 0.150673, 0.173194,
                  1.392306, 0.171382, 0.024487, 0.620108, 0.125562, 0.008054, 0.696872, 4.880167],
    "machine-8": [0.43243, 0.423448, 1.257187, 0.370652, 0.647165, 0.259097, 0.310645, 0.104729,
                  0.206983, 0.295667, 0.253205, 0.092269, 0.037091, 0.423317, 0.072069, 0.092794,
                  0.750663, 0.078322, 0.011033, 0.351416, 0.06762, 0.006779, 0.348711, 2.664105],
    "machine-9": [0.419349, 0.414428, 1.167006, 0.339353, 0.591904, 0.252308, 0.318825, 0.105789,
                  0.199466, 0.294326, 0.264716, 0.102187, 0.035179, 0.396874, 0.07311, 0.092612,
                  0.781863, 0.083899, 0.01139, 0.343647, 0.063564, 0.005972, 0.359388, 2.721527],
    "machine-10": [0.427751, 0.421193, 1.222655, 0.358022, 0.631098, 0.259255, 0.30508, 0.10712,
                   0.194617, 0.299381, 0.267442, 0.111799, 0.038206, 0.412892, 0.062555, 0.088899,
                   0.76876, 0.078724, 0.011722, 0.344909, 0.062131, 0.005796, 0.346868, 2.748466],
    # SERVER 10 → device-2 (edge), device-1/3/4/7 (cloud)
    "device-1": [0.231632, 0.23654, 0.66123, 0.203413, 0.356272, 0.143218, 0.144002, 0.057039,
                 0.089845, 0.16986, 0.105751, 0.035556, 0.00793, 0.187801, 0.038405, 0.042046,
                 0.302779, 0.036955, 0.004167, 0.153534, 0.029828, 0.002031, 0.148579, 1.099664],
    "device-2": [0.231462, 0.224405, 0.664129, 0.203695, 0.354997, 0.145536, 0.146706, 0.060165,
                 0.090015, 0.172522, 0.138587, 0.081288, 0.016344, 0.193128, 0.012082, 0.044079,
                 0.321435, 0.035257, 0.004233, 0.149399, 0.029461, 0.001933, 0.145792, 1.03208],
    "device-3": [0.233049, 0.229084, 0.64088, 0.201833, 0.331652, 0.146534, 0.142117, 0.055378,
                 0.089177, 0.166828, 0.106322, 0.037089, 0.016646, 0.159367, 0.029198, 0.041933,
                 0.365855, 0.03736, 0.004178, 0.143871, 0.029198, 0.001973, 0.14236, 1.190481],
    "device-4": [0.288659, 0.28528, 0.749329, 0.247991, 0.391829, 0.193012, 0.170427, 0.080089,
                 0.119335, 0.228024, 0.15564, 0.06597, 0.020802, 0.222478, 0.049596, 0.054435,
                 0.430497, 0.050546, 0.00507, 0.19489, 0.04217, 0.002161, 0.207606, 1.470272],
    "device-7": [0.496231, 0.502202, 1.337378, 0.505741, 0.659243, 0.363028, 0.296664, 0.15593,
                 0.224405, 0.434188, 0.25782, 0.147856, 0.03466, 0.391023, 0.090304, 0.094345,
                 0.773095, 0.108473, 0.006964, 0.357239, 0.081163, 0.003297, 0.387367, 2.433414],
}


def compute_dmsf_times(layer_times_edge, layer_times_cloud):
    """
    Compute DMSF edge_times and cloud_times dicts for split points {3,5,7,10}.

    edge_times[sp]  = cumulative sum(layers 0..sp) on edge device
    cloud_times[sp] = cumulative sum(layers sp+1..23) on cloud device
                      (backbone sp+1..10 exact; neck/head 11-23 approximate for DMSF)
    """
    edge_times = {}
    cloud_times = {}
    for sp in SPLIT_POINTS:
        edge_times[sp]  = sum(layer_times_edge[:sp + 1])
        cloud_times[sp] = sum(layer_times_cloud[sp + 1:])
    return edge_times, cloud_times


def build_profile(edge_device, cloud_device, batch_size=32):
    """Build profile dict for a given edge/cloud device pair."""
    e_times = LAB_DATA[edge_device]
    c_times = LAB_DATA[cloud_device]
    edge_times, cloud_times = compute_dmsf_times(e_times, c_times)
    return {
        "edge_device": edge_device,
        "cloud_device": cloud_device,
        "batch_size": batch_size,
        "edge_times_ms": edge_times,
        "cloud_times_ms": cloud_times,
        "note": "Preloaded from lab profiling (yolo26n backbone approx for DMSF26n)"
    }


def build_all_profiles(batch_size=32):
    """Build unified profile dict mapping device_name → {edge_times_ms, cloud_times_ms}."""
    all_profiles = {}
    for name, times in LAB_DATA.items():
        edge_times, cloud_times = compute_dmsf_times(times, times)
        all_profiles[name] = {
            "edge_times_ms": {str(sp): t for sp, t in edge_times.items()},
            "cloud_times_ms": {str(sp): t for sp, t in cloud_times.items()},
        }
    return {"batch_size": batch_size, "devices": all_profiles}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--edge', default='machine-2', choices=list(LAB_DATA.keys()),
                   help='Edge device key')
    p.add_argument('--cloud', default='device-1', choices=list(LAB_DATA.keys()),
                   help='Cloud device key')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--out', default=None, help='Output JSON path (default: auto)')
    p.add_argument('--generate-all', action='store_true',
                   help='Generate unified profile for all devices')
    args = p.parse_args()

    if args.generate_all:
        all_prof = build_all_profiles(args.batch)
        out_path = args.out or f"profiles/all_devices_bs{args.batch}.json"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(all_prof, f, indent=2)
        print(f"[Profile] Saved unified profile ({len(all_prof['devices'])} devices): {out_path}")
        return

    profile = build_profile(args.edge, args.cloud, args.batch)
    out_path = args.out or f"profiles/{args.edge}_{args.cloud}_bs{args.batch}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(profile, f, indent=2)

    print(f"[Profile] Saved: {out_path}")
    print(f"\n  Split  | Edge(ms) | Cloud(ms) | Total(ms)")
    print(f"  -------|----------|-----------|----------")
    for sp in SPLIT_POINTS:
        e = profile['edge_times_ms'][str(sp)] if isinstance(list(profile['edge_times_ms'].keys())[0], str) else profile['edge_times_ms'][sp]
        c = profile['cloud_times_ms'][str(sp)] if isinstance(list(profile['cloud_times_ms'].keys())[0], str) else profile['cloud_times_ms'][sp]
        print(f"     {sp:2d}  | {e:8.2f} | {c:9.2f} | {e+c:8.2f}")


if __name__ == '__main__':
    main()
