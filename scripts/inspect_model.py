"""Load the pinned model, check the weights really loaded, and print where the text layers live."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from load_policy import check_weights_loaded, load_config, load_policy

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/smoke.yaml")
out_dir = Path("evidence/setup")
out_dir.mkdir(parents=True, exist_ok=True)

t0 = time.perf_counter()
policy, pre, post, model_dir = load_policy(cfg["pins"], device="cpu")
print(f"load_seconds {time.perf_counter() - t0:.1f}")

report = check_weights_loaded(policy, model_dir)
(out_dir / "weights_check.json").write_text(json.dumps(report, indent=2))
for key, value in report.items():
    print(key, value if isinstance(value, int) else f"{len(value)} {value[:5]}")

vlm = policy.model.vlm
tree = "\n".join(f"{name}: {type(mod).__name__}" for name, mod in vlm.named_modules() if name.count(".") <= 3)
(out_dir / "module_tree.txt").write_text(tree + "\n\n" + repr(vlm))
print(type(vlm).__name__)
print(tree[:3500])
print("n_params", sum(p.numel() for p in vlm.parameters()))
print("vocab", len(policy.model.processor.tokenizer))
