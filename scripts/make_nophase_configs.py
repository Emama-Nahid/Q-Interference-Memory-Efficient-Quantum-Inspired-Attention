from pathlib import Path
import re

CONFIGS = [
    "configs/main_qhybrid.yaml",
    "configs/tinystories_qhybrid.yaml",
    "configs/pile10k_qhybrid.yaml",
    "configs/smallc4_qhybrid.yaml",
]

def make_nophase(src_path: str):
    src = Path(src_path)
    text = src.read_text()

    lines = text.splitlines()
    out = []

    in_experiment = False
    in_model = False
    changed_name = False
    changed_disable_phase = False
    inserted_disable_phase = False

    for line in lines:
        stripped = line.strip()

        # top-level sections
        if re.match(r"^[A-Za-z_].*:\s*$", line):
            in_experiment = stripped == "experiment:"
            in_model = stripped == "model:"

        # experiment.name -> append _nophase
        if in_experiment and re.match(r"^\s*name:\s*", line) and not changed_name:
            m = re.match(r"^(\s*name:\s*)(.+?)\s*$", line)
            if m:
                name = m.group(2).strip()
                if not name.endswith("_nophase"):
                    line = f"{m.group(1)}{name}_nophase"
                changed_name = True

        # model.disable_phase -> true
        if in_model and re.match(r"^\s*disable_phase:\s*", line):
            indent = re.match(r"^(\s*)", line).group(1)
            line = f"{indent}disable_phase: true"
            changed_disable_phase = True

        out.append(line)

        # If disable_phase doesn't exist, insert it right after phase_bound
        if (
            in_model
            and re.match(r"^\s*phase_bound:\s*", line)
            and not changed_disable_phase
            and not inserted_disable_phase
        ):
            indent = re.match(r"^(\s*)", line).group(1)
            out.append(f"{indent}disable_phase: true")
            inserted_disable_phase = True

    # If phase_bound was missing and disable_phase was never inserted, add it before train:
    if not changed_disable_phase and not inserted_disable_phase:
        final = []
        inserted = False
        in_model = False
        for line in out:
            stripped = line.strip()
            if re.match(r"^[A-Za-z_].*:\s*$", line):
                if stripped == "model:":
                    in_model = True
                elif in_model and not inserted:
                    final.append("  disable_phase: true")
                    inserted = True
                    in_model = False
            final.append(line)
        out = final

    dst = src.with_name(src.stem + "_nophase.yaml")
    dst.write_text("\n".join(out) + "\n")
    print(f"Wrote: {dst}")

for cfg in CONFIGS:
    make_nophase(cfg)