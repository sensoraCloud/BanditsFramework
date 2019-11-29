#!/usr/bin/env sh
set -e


TAG=$(git describe --tags `git rev-list --tags --max-count=1`)

echo "{\"VAULT_PASSWORD\": \"$VAULT_PASSWORD\",\"GITHUB_TOKEN\": \"$GITHUB_TOKEN\",\"MODEL_VERSION\": \"$TAG\"}" > ../tag-code/build-args.json

echo "$TAG" > ../tag-code/image-tag

cat ../tag-code/image-tag
