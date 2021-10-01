#!/usr/bin/bash

# Set up ssh for private git repos
eval "$(ssh-agent)"
ssh-add ~/.ssh/id_ed25519

# Main command
sudo DOCKER_BUILDKIT=1 docker build --ssh default="${SSH_AUTH_SOCK}" .
