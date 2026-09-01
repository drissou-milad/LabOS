# Run09 file-placement verification script.
# Run from: C:\Users\acer\Desktop\LabOS
# Usage:    .\verify_run09_files.ps1
#
# Checks only for FILE EXISTENCE (not content/correctness). Exits with a summary of
# FOUND / MISSING for every file Run09 needs, split into three groups: the new Run09 files
# you are about to copy, the existing dependency this package requires but does not include,
# and the pre-existing LabOS/Run08 files Run09 reads from but never modifies.

$ErrorActionPreference = "Continue"

function Check-Path($relativePath, $label) {
    if (Test-Path $relativePath) {
        Write-Host "FOUND    $label -> $relativePath" -ForegroundColor Green
        return $true
    } else {
        Write-Host "MISSING  $label -> $relativePath" -ForegroundColor Red
        return $false
    }
}

Write-Host "=================================================================="
Write-Host "GROUP 1: NEW Run09 files (from this package -- copy these in first)"
Write-Host "=================================================================="
$group1 = @(
    @{Path="scripts\benchmark_v2\train_exp05_run09.py"; Label="Run09 training script"},
    @{Path="scripts\benchmark_v2\diagnose_run09_test.py"; Label="Run09 test evaluation script"},
    @{Path="scripts\benchmark_v2\diagnose_matched_contrast.py"; Label="Matched-contrast diagnostic (Run08+Run09)"}
)
$group1Results = $group1 | ForEach-Object { Check-Path $_.Path $_.Label }

Write-Host ""
Write-Host "=================================================================="
Write-Host "GROUP 2: Existing dependency (must ALREADY be in your project --"
Write-Host "         this package does not include or modify it)"
Write-Host "=================================================================="
$group2 = @(
    @{Path="scripts\benchmark_v2\diagnose_cross_sample_shift.py"; Label="Required by diagnose_matched_contrast.py (import)"}
)
$group2Results = $group2 | ForEach-Object { Check-Path $_.Path $_.Label }

Write-Host ""
Write-Host "=================================================================="
Write-Host "GROUP 3: Existing LabOS/src files Run09 reads (unmodified imports)"
Write-Host "=================================================================="
$group3 = @(
    @{Path="src\model.py"; Label="CellCNN architecture"},
    @{Path="src\dataset.py"; Label="(not imported by Run09, listed for reference only)"},
    @{Path="src\augmentations.py"; Label="augment_patch (flip/rotate/brightness/noise)"},
    @{Path="src\losses.py"; Label="FocalLoss + get_criterion"},
    @{Path="src\train.py"; Label="validate, collect_predictions"},
    @{Path="src\experiment.py"; Label="Experiment (run tracking)"},
    @{Path="src\config.py"; Label="(not imported directly by Run09, listed for reference only)"}
)
$group3Results = $group3 | ForEach-Object { Check-Path $_.Path $_.Label }

Write-Host ""
Write-Host "=================================================================="
Write-Host "GROUP 4: Existing Run08 dataset files Run09 reads (never modifies)"
Write-Host "=================================================================="
$group4 = @(
    @{Path="results\exp05_training_dataset\run02_20samples\manifest.csv"; Label="manifest.csv"},
    @{Path="results\exp05_training_dataset\run02_20samples\split_manifest.csv"; Label="split_manifest.csv (frozen split)"},
    @{Path="results\exp05_training_dataset\run02_20samples\patches"; Label="patches directory"}
)
$group4Results = $group4 | ForEach-Object { Check-Path $_.Path $_.Label }

Write-Host ""
Write-Host "=================================================================="
Write-Host "GROUP 5: Run08 output (informational -- Run09 does NOT touch this;"
Write-Host "         listed only so you can confirm it still exists untouched)"
Write-Host "=================================================================="
$group5 = @(
    @{Path="results\exp05_training_dataset\run02_20samples\run08_patch_instance_norm\test_predictions_resumable.csv"; Label="Run08 test predictions"}
)
$group5Results = $group5 | ForEach-Object { Check-Path $_.Path $_.Label }

Write-Host ""
Write-Host "=================================================================="
Write-Host "GROUP 6: Run09 output directory collision check"
Write-Host "=================================================================="
$run09OutDir = "results\exp05_training_dataset\run02_20samples\run09_contrast_aug_focal_robust_eps"
if (Test-Path $run09OutDir) {
    Write-Host "WARNING  $run09OutDir already exists -- train_exp05_run09.py will refuse to" -ForegroundColor Yellow
    Write-Host "         write into it unless empty or --allow-nonempty-out-dir is passed." -ForegroundColor Yellow
} else {
    Write-Host "OK       $run09OutDir does not exist yet (will be created fresh by training)." -ForegroundColor Green
}

Write-Host ""
Write-Host "=================================================================="
Write-Host "GROUP 7: Python package check"
Write-Host "=================================================================="
python -c "import torch, numpy, sklearn, scipy; print('OK: torch', torch.__version__, '| numpy', numpy.__version__, '| sklearn', sklearn.__version__, '| scipy', scipy.__version__)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "MISSING  one or more required Python packages (torch, numpy, scikit-learn, scipy)" -ForegroundColor Red
}

Write-Host ""
$allGroup1 = ($group1Results -notcontains $false)
$allGroup2 = ($group2Results -notcontains $false)
$allGroup3 = ($group3Results -notcontains $false)
$allGroup4 = ($group4Results -notcontains $false)
Write-Host "=================================================================="
if ($allGroup1 -and $allGroup2 -and $allGroup3 -and $allGroup4) {
    Write-Host "SUMMARY: All required files found. Proceed to the PRE-RUN CHECKLIST." -ForegroundColor Green
} else {
    Write-Host "SUMMARY: One or more required files are MISSING (see above). Do not launch training yet." -ForegroundColor Red
}
Write-Host "=================================================================="
