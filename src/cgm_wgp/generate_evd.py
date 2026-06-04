"""
generate_evd.py

Generate EVD.rda (eigenvalue decomposition) from a G-matrix file (.rds or .rda)
or from a G-matrix stored as CSV (when computed in Python from markers).

Example:
  venv/bin/python3 scripts/generate_evd.py \
    --gmatrix pipeline/2_gmatrix/Beans/Gmatrix.rda \
    --output pipeline/2_gmatrix/Beans/EVD.rda
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_rscript() -> str | None:
    """Locate Rscript binary."""
    p = shutil.which("Rscript")
    if p:
        return p
    for c in ["/opt/homebrew/bin/Rscript", "/usr/local/bin/Rscript", "/usr/bin/Rscript"]:
        if Path(c).is_file():
            return c
    return None


def _run_r_code(rscript: str, r_code: str) -> bool:
    """Run R code via Rscript subprocess. Returns True on success."""
    result = subprocess.run(
        [rscript, "-e", r_code],
        capture_output=True, text=True, timeout=300,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            if line.strip():
                print(f"  [R] {line}")

    if result.returncode != 0:
        print(f"\nError: R subprocess failed (exit {result.returncode})", file=sys.stderr)
        return False
    return True


def generate_evd_from_gmatrix(
    gmatrix_path: str,
    output_path: str,
    save_gmatrix: bool = False,
    rscript_path: str | None = None,
) -> bool:
    """Generate EVD.rda from a G-matrix file (.rds or .rda).

    Returns True on success.
    """
    rscript = rscript_path or find_rscript()
    if rscript is None:
        print("Error: Rscript not found. Install R or provide rscript_path.", file=sys.stderr)
        return False

    gmatrix_abs = str(Path(gmatrix_path).resolve())
    if not Path(gmatrix_abs).exists():
        print(f"Error: G-matrix file not found: {gmatrix_abs}", file=sys.stderr)
        return False

    output_abs = str(Path(output_path).resolve())
    Path(output_abs).parent.mkdir(parents=True, exist_ok=True)

    ext = Path(gmatrix_abs).suffix.lower()

    if ext == ".rds":
        load_code = f'G <- readRDS("{gmatrix_abs}")'
    elif ext == ".rda":
        load_code = f"""
load("{gmatrix_abs}")
# Find the G-matrix variable (first matrix-like object)
all_vars <- ls()
G <- NULL
for (vname in all_vars) {{
  obj <- get(vname)
  if (is.matrix(obj) && nrow(obj) == ncol(obj)) {{
    G <- obj
    cat("Found G-matrix in variable:", vname, "\\n")
    break
  }}
}}
if (is.null(G)) stop("No square matrix found in .rda file")
"""
    else:
        print(f"Error: Unsupported format '{ext}'. Use .rds or .rda", file=sys.stderr)
        return False

    save_gmatrix_code = ""
    if save_gmatrix:
        gmatrix_rda = str(Path(output_abs).parent / "Gmatrix.rda")
        save_gmatrix_code = f'save(G, file="{gmatrix_rda}")\ncat("Saved Gmatrix.rda\\n")'

    r_code = _build_evd_r_code(load_code, output_abs, save_gmatrix_code)

    print(f"Loading G-matrix from: {gmatrix_abs}")
    print(f"Output EVD.rda to: {output_abs}")
    print()

    success = _run_r_code(rscript, r_code)
    if success:
        print("Done.")
    return success


def generate_evd_from_csv(
    csv_path: str,
    genotype_ids: list[str],
    output_path: str,
    save_gmatrix: bool = True,
    rscript_path: str | None = None,
) -> bool:
    """Generate EVD.rda + Gmatrix.rda from a G-matrix stored as CSV.

    Used when Python computes G from markers and saves as CSV.
    The CSV should be a square matrix with no header/index.
    genotype_ids provides the row/column names.

    Returns True on success.
    """
    rscript = rscript_path or find_rscript()
    if rscript is None:
        print("Error: Rscript not found. Install R or provide rscript_path.", file=sys.stderr)
        return False

    csv_abs = str(Path(csv_path).resolve())
    if not Path(csv_abs).exists():
        print(f"Error: G-matrix CSV not found: {csv_abs}", file=sys.stderr)
        return False

    output_abs = str(Path(output_path).resolve())
    Path(output_abs).parent.mkdir(parents=True, exist_ok=True)

    # Write genotype IDs to a temp file for R to read
    ids_path = str(Path(output_abs).parent / "_genotype_ids.txt")
    with open(ids_path, "w") as f:
        for gid in genotype_ids:
            f.write(f"{gid}\n")

    load_code = f"""
G <- as.matrix(read.csv("{csv_abs}", header=FALSE))
ids <- readLines("{ids_path}")
rownames(G) <- ids
colnames(G) <- ids
"""

    save_gmatrix_code = ""
    if save_gmatrix:
        gmatrix_rda = str(Path(output_abs).parent / "Gmatrix.rda")
        save_gmatrix_code = f'save(G, file="{gmatrix_rda}")\ncat("Saved Gmatrix.rda\\n")'

    r_code = _build_evd_r_code(load_code, output_abs, save_gmatrix_code)

    print(f"Loading G-matrix CSV from: {csv_abs}")
    print(f"Output EVD.rda to: {output_abs}")
    print()

    success = _run_r_code(rscript, r_code)

    # Clean up temp file
    Path(ids_path).unlink(missing_ok=True)

    if success:
        print("Done.")
    return success


def _build_evd_r_code(load_code: str, output_path: str, save_gmatrix_code: str) -> str:
    """Build the R code for eigendecomposition."""
    return f"""
{load_code}

cat("G-matrix dimensions:", nrow(G), "x", ncol(G), "\\n")

# Check for row/column names
ids <- rownames(G)
if (is.null(ids)) {{
  ids <- colnames(G)
}}
if (is.null(ids)) {{
  cat("Warning: No row/column names found, using G1..Gn\\n")
  ids <- paste0("G", seq_len(nrow(G)))
}}
cat("Genotype IDs (first 5):", paste(head(ids, 5), collapse=", "), "\\n")
cat("Total genotypes:", length(ids), "\\n")

# Compute eigenvalue decomposition
cat("Computing eigendecomposition...\\n")
EVD <- eigen(G)

# Set rownames on eigenvectors to match genotype IDs
rownames(EVD$vectors) <- ids

# Summary
cat("Top 5 eigenvalues:", paste(round(head(EVD$values, 5), 4), collapse=", "), "\\n")
n_pos <- sum(EVD$values > 1e-10)
cat("Positive eigenvalues:", n_pos, "of", length(EVD$values), "\\n")
var_explained_5 <- sum(head(EVD$values, 5)) / sum(EVD$values[EVD$values > 0]) * 100
cat("Variance explained by top 5:", round(var_explained_5, 1), "%\\n")

# Save EVD
save(EVD, file="{output_path}")
cat("\\nSaved EVD.rda to: {output_path}\\n")

{save_gmatrix_code}
"""


def main():
    p = argparse.ArgumentParser(description="Generate EVD.rda from a G-matrix file.")
    p.add_argument("--gmatrix", required=True, help="Path to G-matrix file (.rds or .rda)")
    p.add_argument("--output", required=True, help="Output path for EVD.rda")
    p.add_argument("--save-gmatrix", action="store_true",
                   help="Also save the G-matrix as Gmatrix.rda alongside the EVD")
    p.add_argument("--rscript", default=None, help="Path to Rscript binary (auto-detected if omitted)")
    args = p.parse_args()

    success = generate_evd_from_gmatrix(
        gmatrix_path=args.gmatrix,
        output_path=args.output,
        save_gmatrix=args.save_gmatrix,
        rscript_path=args.rscript,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
