#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python3}
mkdir -p "$project_dir/.build-tmp"
export TMPDIR="$project_dir/.build-tmp"
if [[ "${D4CNN_DIRECT_BUILD:-0}" == "1" ]]; then
  exec "$python_bin" "$project_dir/build_direct.py"
fi
pybind_dir=$("$python_bin" -m pybind11 --cmakedir)
cmake -S "$project_dir" -B "$project_dir/build/local" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="${CPU_CXX:-g++}" \
  -DPython_EXECUTABLE="$python_bin" -Dpybind11_DIR="$pybind_dir" \
  -DANACAL_SOURCE_DIR="${ANACAL_SOURCE_DIR:-}"
cmake --build "$project_dir/build/local" -j "${BUILD_JOBS:-2}"
# Place the extension beside the package for python -m cpu_src from repo root.
extension_suffix=$("$python_bin" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
cp "$project_dir/build/local/_native$extension_suffix" "$project_dir/.build-tmp/_native$extension_suffix"
mv -f "$project_dir/.build-tmp/_native$extension_suffix" "$project_dir/_native$extension_suffix"
