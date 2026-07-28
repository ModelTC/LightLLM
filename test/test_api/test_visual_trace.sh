#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

API_URL=${API_URL:-http://127.0.0.1:8000/v1/chat/completions}
MODEL=${MODEL:-main_agent}
IMAGE_PATH=${IMAGE_PATH:-${SCRIPT_DIR}/test.jpg}
TRACE_DIR=${TRACE_DIR:-$(dirname "${REPO_ROOT}")/visual_trajectories}

if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "Image not found: ${IMAGE_PATH}" >&2
    exit 1
fi

case "${IMAGE_PATH,,}" in
    *.png) MIME_TYPE=image/png ;;
    *.webp) MIME_TYPE=image/webp ;;
    *.gif) MIME_TYPE=image/gif ;;
    *) MIME_TYPE=image/jpeg ;;
esac

IMAGE_BASE64=$(base64 "${IMAGE_PATH}" | tr -d '\n')
MARKER_FILE=$(mktemp)
RESPONSE_FILE=$(mktemp)
trap 'rm -f "${MARKER_FILE}" "${RESPONSE_FILE}"' EXIT

echo "API: ${API_URL}"
echo "Image: ${IMAGE_PATH}"
echo "Trace directory: ${TRACE_DIR}"

HTTP_STATUS=$(
    curl --silent --show-error --max-time 300 \
        --output "${RESPONSE_FILE}" \
        --write-out '%{http_code}' \
        --header 'Content-Type: application/json' \
        --data-binary @- \
        "${API_URL}" <<JSON
{
  "model": "${MODEL}",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "请仔细查看图片，描述图片中的主要内容，并指出最显眼物体的颜色。"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:${MIME_TYPE};base64,${IMAGE_BASE64}"
          }
        }
      ]
    }
  ],
  "temperature": 0,
  "max_tokens": 512,
  "stream": false
}
JSON
)

echo "HTTP status: ${HTTP_STATUS}"
python -m json.tool "${RESPONSE_FILE}" 2>/dev/null || cat "${RESPONSE_FILE}"

mapfile -t TRACE_FILES < <(
    find "${TRACE_DIR}" -maxdepth 1 -type f -name '*.json' -newer "${MARKER_FILE}" -print 2>/dev/null | sort
)

if [[ ${#TRACE_FILES[@]} -eq 0 ]]; then
    echo "No new trace file found. Check that LIGHTLLM_VISUAL_TRACE_DUMP=1 was present when the service started." >&2
else
    echo "New trace files:"
    printf '  %s\n' "${TRACE_FILES[@]}"
    python - "${TRACE_FILES[-1]}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    trace = json.load(handle)

print(f"Trace ID: {trace.get('trace_id')}")
print(f"Status: {trace.get('status')}")
print("Events:")
for event in trace.get("events", []):
    step = event.get("step")
    suffix = f" step={step}" if step is not None else ""
    print(f"  {event.get('index')}: {event.get('type')}{suffix}")
PY
fi

if [[ ! "${HTTP_STATUS}" =~ ^2 ]]; then
    exit 1
fi
