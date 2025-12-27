#!/usr/bin/env bash
set -euo pipefail

# Resumable asset downloader for iwoa-nas.
# Stores everything under ./data (already gitignored in this repo).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p data/datasets data/benchmarks data/checkpoints/ofa data/logs

echo "[bootstrap_download_assets] root=$ROOT_DIR"
echo "[bootstrap_download_assets] started=$(date -Is)"

###############################################################################
# 1) CIFAR-100-C (Zenodo)
###############################################################################
# Official Zenodo record: 3555552 (CIFAR-100-C)
CIFAR100C_DIR="data/datasets/cifar-100-c"
CIFAR100C_TAR="$CIFAR100C_DIR/CIFAR-100-C.tar"
CIFAR100C_URL="https://zenodo.org/api/records/3555552/files/CIFAR-100-C.tar/content"
CIFAR100C_EXPECTED_BYTES="2918473216"

mkdir -p "$CIFAR100C_DIR"

need_cifar100c_download="true"
if [[ -f "$CIFAR100C_TAR" ]]; then
  size_bytes="$(stat -c%s "$CIFAR100C_TAR" || echo 0)"
  if [[ "$size_bytes" -ge "$CIFAR100C_EXPECTED_BYTES" ]]; then
    need_cifar100c_download="false"
    echo "[cifar-100-c] tar already present (size=${size_bytes}B): $CIFAR100C_TAR"
  else
    echo "[cifar-100-c] tar is partial (size=${size_bytes}B < ${CIFAR100C_EXPECTED_BYTES}B) — will resume"
  fi
fi

if [[ "$need_cifar100c_download" == "true" ]]; then
  echo "[cifar-100-c] downloading (resumable): $CIFAR100C_URL"
  curl -L --fail --retry 10 --retry-delay 5 -C - -o "$CIFAR100C_TAR" "$CIFAR100C_URL"
  size_bytes="$(stat -c%s "$CIFAR100C_TAR" || echo 0)"
  if [[ "$size_bytes" -lt "$CIFAR100C_EXPECTED_BYTES" ]]; then
    echo "[cifar-100-c] ERROR: download incomplete after resume (size=${size_bytes}B < ${CIFAR100C_EXPECTED_BYTES}B)" >&2
    exit 1
  fi
fi

if [[ -d "$CIFAR100C_DIR/CIFAR-100-C" ]]; then
  echo "[cifar-100-c] already extracted: $CIFAR100C_DIR/CIFAR-100-C"
else
  echo "[cifar-100-c] extracting..."
  tar -xf "$CIFAR100C_TAR" -C "$CIFAR100C_DIR"
fi

###############################################################################
# 2) NAS-Bench-201 benchmark file (Google Drive)
###############################################################################
# Official link (from NAS-Bench-201 README):
# https://drive.google.com/open?id=16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_
NASB201_ID="16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_"
NASB201_OUT="data/benchmarks/NAS-Bench-201-v1_1-096897.pth"

if [[ -f "$NASB201_OUT" ]]; then
  echo "[nasbench201] already present: $NASB201_OUT"
else
  echo "[nasbench201] downloading (resumable) -> $NASB201_OUT"
  source .venv/bin/activate
  python -m gdown --continue --id "$NASB201_ID" -O "$NASB201_OUT"
fi

###############################################################################
# 3) OFA MobileNetV3 supernet weights
###############################################################################
# The once-for-all code downloads these from han-cai/files:
# https://raw.githubusercontent.com/han-cai/files/master/ofa/ofa_nets/<net_id>
OFA_NET_ID="ofa_mbv3_d234_e346_k357_w1.0"
OFA_URL="https://raw.githubusercontent.com/han-cai/files/master/ofa/ofa_nets/${OFA_NET_ID}"
OFA_OUT="data/checkpoints/ofa/${OFA_NET_ID}.pth"

if [[ -f "$OFA_OUT" ]]; then
  echo "[ofa] already present: $OFA_OUT"
else
  echo "[ofa] downloading (resumable) -> $OFA_OUT"
  curl -L --fail --retry 10 --retry-delay 5 -C - -o "$OFA_OUT" "$OFA_URL"
fi

echo "[bootstrap_download_assets] finished=$(date -Is)"

