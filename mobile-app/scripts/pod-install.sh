#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

BUNDLE_PATH=.bundle/vendor bundle install

mkdir -p ios
printf 'export NODE_BINARY=%s\n' "$(node -p 'process.execPath')" > ios/.xcode.env.local

if ! command -v xcodebuild >/dev/null 2>&1; then
  echo "xcodebuild is required. Run pod installation on macOS with Xcode installed." >&2
  exit 1
fi

cd ios
RCT_NEW_ARCH_ENABLED=1 BUNDLE_PATH=../.bundle/vendor bundle exec pod install
