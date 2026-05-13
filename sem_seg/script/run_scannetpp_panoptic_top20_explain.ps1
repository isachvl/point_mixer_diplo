param(
    [string]$SceneId = "00a231a370",
    [string]$ScenePly = "",
    [string]$Checkpoint = "/workspace/outputs/PointMixerScanNetPP_panoptic_rgb_pv004_block_balanced/2026-05-02_19-40-27__scannetpp__pointmixer_panoptic_3060/epoch=020--mIoU_val=0.1720--.ckpt",
    [string]$DatasetRoot = "/datasets/pointmixer_scannetpp_top100_panoptic_rgb_pv004",
    [string]$OutputRoot = "/workspace/outputs/pointmixer_top20_explain",
    [int]$TopK = 20,
    [double]$InputVoxelSize = 0.04,
    [int]$MaxPoints = 16000,
    [int]$MinPoints = 1024,
    [double]$BlockSize = 2.5,
    [int]$Votes = 1,
    [double]$ConfidenceThreshold = 0.15,
    [double]$ClusterRadius = 0.12,
    [int]$MinClusterPoints = 50,
    [switch]$WritePointCsv
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ScenePly)) {
    $ScenePly = "/datasets/scannetpp_full/data/$SceneId/scans/mesh_aligned_0.05.ply"
}

$Stamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$RunDir = "$OutputRoot/${SceneId}_top${TopK}_$Stamp"
$LabelMap = "$DatasetRoot/meta/label_mapping.tsv"
$ThingClasses = "$DatasetRoot/meta/thing_classes.txt"
$ExplainLog = "$RunDir/${SceneId}_top${TopK}_explain.txt"
$ConsoleLog = "$RunDir/${SceneId}_console.log"
$PointCsvArg = ""
if ($WritePointCsv) {
    $PointCsvArg = "--write-point-csv"
}

$Bash = @"
set -euo pipefail
mkdir -p '$RunDir'
cd /code/ECCV22-PointMixer/sem_seg

if [ ! -f '$Checkpoint' ]; then
  echo '[PM TOP20 ERROR] checkpoint not found: $Checkpoint'
  echo '[PM TOP20 HINT] If you only have the backup in E:\saves, use the original /workspace/outputs checkpoint or copy it into a Docker-mounted folder.'
  exit 2
fi
if [ ! -f '$ScenePly' ]; then
  echo '[PM TOP20 ERROR] scene ply not found: $ScenePly'
  exit 3
fi
if [ ! -f '$LabelMap' ]; then
  echo '[PM TOP20 ERROR] label map not found: $LabelMap'
  exit 4
fi

python tools/infer_scannetpp_panoptic_scene.py \
  --scene-ply '$ScenePly' \
  --checkpoint '$Checkpoint' \
  --label-map '$LabelMap' \
  --thing-classes '$ThingClasses' \
  --output-dir '$RunDir' \
  --input-voxel-size '$InputVoxelSize' \
  --max-points '$MaxPoints' \
  --min-points '$MinPoints' \
  --block-size '$BlockSize' \
  --votes '$Votes' \
  --confidence-threshold '$ConfidenceThreshold' \
  --cluster-radius '$ClusterRadius' \
  --min-cluster-points '$MinClusterPoints' \
  --top-k-log '$TopK' \
  --explain-log '$ExplainLog' \
  $PointCsvArg \
  2>&1 | tee '$ConsoleLog'
"@

Write-Host "[PM TOP20] scene: $ScenePly"
Write-Host "[PM TOP20] checkpoint: $Checkpoint"
Write-Host "[PM TOP20] output: $RunDir"
docker exec pointmixer-scannetpp bash -lc $Bash

$WindowsRunDir = $RunDir -replace "^/workspace/outputs", "E:\pointmixer_outputs"
Write-Host "[PM TOP20] Windows output folder: $WindowsRunDir"
Write-Host "[PM TOP20] Explain log: $($ExplainLog -replace '^/workspace/outputs', 'E:\pointmixer_outputs')"
