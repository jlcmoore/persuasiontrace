#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_DEST="/private/tmp/continuouspersuasion-review"

DEST="${1:-${DEFAULT_DEST}}"
EXCLUDES_FILE="${2:-${SCRIPT_DIR}/review_export_excludes.txt}"
RESULTS_KEEP_FILE="${3:-${SCRIPT_DIR}/review_export_results_keep.txt}"
EXPORT_DEST=""
EXPORT_DEST_IS_TEMP=0
UPDATE_EXISTING_GIT_REPO=0

if [[ ! -f "${EXCLUDES_FILE}" ]]; then
  echo "Exclusion file not found: ${EXCLUDES_FILE}" >&2
  exit 1
fi

if [[ -e "${DEST}" ]]; then
  if git -C "${DEST}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    UPDATE_EXISTING_GIT_REPO=1
    EXPORT_DEST="$(mktemp -d)"
    EXPORT_DEST_IS_TEMP=1
  else
    echo "Destination exists and is not a git repository: ${DEST}" >&2
    echo "Remove it or pass a different destination path." >&2
    exit 1
  fi
else
  mkdir -p "${DEST}"
  EXPORT_DEST="${DEST}"
fi

TRACKED_FILE_LIST="$(mktemp)"
PRUNED_ROOT=""
cleanup() {
  rm -f "${TRACKED_FILE_LIST}"
  if [[ -n "${PRUNED_ROOT}" ]]; then
    rm -rf "${PRUNED_ROOT}"
  fi
  if [[ "${EXPORT_DEST_IS_TEMP}" -eq 1 && -n "${EXPORT_DEST}" ]]; then
    rm -rf "${EXPORT_DEST}"
  fi
}
trap cleanup EXIT

# Copy exactly the tracked paths from the current working tree.
while IFS= read -r -d '' tracked_path; do
  if [[ -e "${REPO_ROOT}/${tracked_path}" ]]; then
    printf '%s\0' "${tracked_path}" >> "${TRACKED_FILE_LIST}"
  fi
done < <(git -C "${REPO_ROOT}" ls-files -z)
rsync -a --from0 --files-from="${TRACKED_FILE_LIST}" "${REPO_ROOT}/" "${EXPORT_DEST}/"

while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
  line_without_comment="${raw_line%%#*}"
  exclude_path="$(printf '%s' "${line_without_comment}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

  if [[ -z "${exclude_path}" ]]; then
    continue
  fi

  if [[ "${exclude_path}" == *"*"* || "${exclude_path}" == *"?"* || "${exclude_path}" == *"["* ]]; then
    while IFS= read -r match_path; do
      if [[ -z "${match_path}" ]]; then
        continue
      fi
      rm -rf "${EXPORT_DEST}/${match_path}"
    done < <(
      (
        cd "${EXPORT_DEST}"
        compgen -G "${exclude_path}" || true
      )
    )
  else
    rm -rf "${EXPORT_DEST}/${exclude_path}"
  fi
done < "${EXCLUDES_FILE}"

if [[ -d "${EXPORT_DEST}/results" && -f "${RESULTS_KEEP_FILE}" ]]; then
  PRUNED_ROOT="$(mktemp -d)"

  mkdir -p "${PRUNED_ROOT}/results"

  while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    line_without_comment="${raw_line%%#*}"
    keep_pattern="$(printf '%s' "${line_without_comment}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    if [[ -z "${keep_pattern}" ]]; then
      continue
    fi

    while IFS= read -r match_path; do
      if [[ -z "${match_path}" ]]; then
        continue
      fi
      (
        cd "${EXPORT_DEST}"
        rsync -a --relative "${match_path}" "${PRUNED_ROOT}/"
      )
    done < <(
      (
        cd "${EXPORT_DEST}"
        compgen -G "${keep_pattern}" || true
      )
    )
  done < "${RESULTS_KEEP_FILE}"

  rm -rf "${EXPORT_DEST}/results"
  mv "${PRUNED_ROOT}/results" "${EXPORT_DEST}/results"
  rm -rf "${PRUNED_ROOT}"
  PRUNED_ROOT=""
fi

if [[ "${UPDATE_EXISTING_GIT_REPO}" -eq 1 ]]; then
  rsync -a --delete --exclude ".git" --exclude ".git/" "${EXPORT_DEST}/" "${DEST}/"
  echo "Updated existing git destination at ${DEST}"
else
  echo "Created review copy at ${DEST}"
fi
echo "Excluded paths from ${EXCLUDES_FILE}"
if [[ -f "${RESULTS_KEEP_FILE}" ]]; then
  echo "Kept results artifacts from ${RESULTS_KEEP_FILE}"
fi
