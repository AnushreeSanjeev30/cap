"""
fix_pickle.py
Run this ONCE to fix the pickle module reference error.
Then run phase_d_multi_model.py to get the full ensemble.
"""
import pickle
import sys
import os

# ── Step 1: Patch __main__ so the old pickle can deserialize ─────────────────
from archive.softpower_models import SoftPowerXGB, SoftPowerRF, SoftPowerLGBM, SoftPowerEnsemble

# Make these classes findable under __main__ (where they were originally pickled)
sys.modules['__main__'].SoftPowerXGB  = SoftPowerXGB
sys.modules['__main__'].SoftPowerRF   = SoftPowerRF
sys.modules['__main__'].SoftPowerLGBM = SoftPowerLGBM

# ── Step 2: Try all known pickle locations ────────────────────────────────────
candidates = [
    'output/artifacts/xgb_model.pkl',
    'output/xgb_model.pkl',
    'artifacts/xgb_model.pkl',
    'xgb_model.pkl',
]

loaded_path = None
model = None

for path in candidates:
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            loaded_path = path
            print(f"✅ Loaded from: {path}")
            print(f"   Type: {type(model)}")
            break
        except Exception as e:
            print(f"❌ Failed {path}: {e}")

if model is None:
    print("\n❌ No model found. Run phase_c_model_ready.py first.")
    sys.exit(1)

# ── Step 3: Re-save with correct module reference ─────────────────────────────
# Ensure the class is now from softpower_models, not __main__
if type(model).__module__ == '__main__':
    # Rebuild as proper SoftPowerXGB instance
    new_model = SoftPowerXGB()
    new_model.models       = model.models
    new_model.feature_cols = model.feature_cols
    new_model.params       = model.params
    model = new_model
    print("   Rebuilt as softpower_models.SoftPowerXGB")

os.makedirs('output/artifacts', exist_ok=True)

# Save to all canonical locations
save_paths = [
    'output/artifacts/xgb_model.pkl',
    'output/best_model.pkl',
]

for path in save_paths:
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Re-saved → {path}")

# ── Step 4: Verify it loads cleanly ──────────────────────────────────────────
print("\nVerifying clean load...")
for path in save_paths:
    with open(path, 'rb') as f:
        m = pickle.load(f)
    print(f"  ✅ {path} → {type(m).__module__}.{type(m).__name__}")
    if hasattr(m, 'feature_cols') and m.feature_cols:
        print(f"     Features: {len(m.feature_cols)} columns")
    if hasattr(m, 'models'):
        print(f"     Quantiles: {list(m.models.keys())}")

print("\n✅ Fix complete.")
print("Next steps:")
print("  1. python phase_d_multi_model.py   ← trains RF + LGBM, builds ensemble")
print("  2. python phase3_causal_complete.py --track A")
