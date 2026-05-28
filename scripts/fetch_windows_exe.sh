#!/usr/bin/env bash
# Запускает сборку .exe в GitHub Actions и скачивает артефакт в dist/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WORKFLOW="build-windows.yml"
ARTIFACT="moe-optimizator-windows"
OUT="$ROOT/dist/moe-optimizator.exe"
REF="${1:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"

if ! command -v gh >/dev/null 2>&1; then
  echo "Нужен GitHub CLI: brew install gh && gh auth login" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "Выполните: gh auth login" >&2
  exit 1
fi

if ! gh repo view >/dev/null 2>&1; then
  echo "Нет доступа к GitHub-репозиторию. Нужен push в origin и gh auth login." >&2
  exit 1
fi

echo "==> Репозиторий: $(gh repo view --json nameWithOwner -q .nameWithOwner)"
echo "==> Запуск workflow $WORKFLOW (ветка: $REF)…"

before_id="$(gh run list --workflow="$WORKFLOW" --limit 1 --json databaseId -q '.[0].databaseId' 2>/dev/null || true)"

gh workflow run "$WORKFLOW" --ref "$REF"

echo "==> Ожидание нового run…"
run_id=""
for _ in $(seq 1 40); do
  sleep 3
  run_id="$(gh run list --workflow="$WORKFLOW" --limit 1 --json databaseId -q '.[0].databaseId')"
  if [[ -n "$run_id" && "$run_id" != "$before_id" ]]; then
    break
  fi
done

if [[ -z "$run_id" || "$run_id" == "$before_id" ]]; then
  echo "Не появился новый workflow run. Проверьте Actions в браузере." >&2
  exit 1
fi

echo "==> Run: https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/actions/runs/$run_id"
gh run watch "$run_id"

status="$(gh run view "$run_id" --json conclusion -q .conclusion)"
if [[ "$status" != "success" ]]; then
  echo "Сборка завершилась с ошибкой: $status" >&2
  gh run view "$run_id" --log-failed >&2 || true
  exit 1
fi

mkdir -p dist
rm -f "$OUT"
echo "==> Скачивание артефакта $ARTIFACT…"
gh run download "$run_id" -n "$ARTIFACT" -D dist

if [[ -f dist/moe-optimizator.exe ]]; then
  :
elif [[ -f dist/moe-optimizator-windows/moe-optimizator.exe ]]; then
  mv dist/moe-optimizator-windows/moe-optimizator.exe "$OUT"
  rmdir dist/moe-optimizator-windows 2>/dev/null || true
else
  found="$(find dist -name 'moe-optimizator.exe' -print -quit 2>/dev/null || true)"
  if [[ -n "$found" ]]; then
    mv "$found" "$OUT"
  else
    echo "Не найден moe-optimizator.exe в артефакте" >&2
    exit 1
  fi
fi

echo "==> Готово: $OUT"
ls -lh "$OUT"
