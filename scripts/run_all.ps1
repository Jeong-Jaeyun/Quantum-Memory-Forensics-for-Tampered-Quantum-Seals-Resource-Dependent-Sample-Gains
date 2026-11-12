# Rerun every experiment config in the original order (about 35 h with 6 workers on 8 cores).
# The results go to a new directory, so the stored rows in outputs/results are never appended to:
#   powershell -File scripts/run_all.ps1 -Results outputs/rerun
# One config: python -m qmf.run configs/e14.json --results outputs/rerun (see README, "Rerun the experiments").
param([string]$Results = "outputs/rerun")
Set-Location (Split-Path -Parent $PSScriptRoot)  # repository root
$env:PYTHONPATH = "src"
# E1f replays the E1 rows and E14 reads e1f.json, both from the same results directory.
foreach ($c in "e1_e3", "e3b", "e5_e8", "pl", "e2_ghz", "e5b", "e9", "e10", "e1f", "e11", "e12", "e13", "e14", "e15") {
    python -m qmf.run "configs/$c.json" --results $Results
    if ($LASTEXITCODE) { throw "configs/$c.json failed (exit $LASTEXITCODE)" }
}
