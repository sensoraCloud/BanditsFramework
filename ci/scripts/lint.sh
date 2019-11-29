#!/usr/bin/env sh
set -ev

TAG=$(git describe --tags `git rev-list --tags --max-count=1`)

echo "Build TAG=$TAG"
git checkout tags/$TAG -b topic/local-deploy

HEAD=$(git rev-parse HEAD)
echo "Current HEAD=$HEAD"

make lint
