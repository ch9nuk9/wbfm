#!/usr/bin/bash

# Set up ssh for private git repos
eval "$(ssh-agent)"
ssh-add ~/.ssh/id_ed25519

echo "Found ssh keys: $(ssh-add -L)"
echo "$(ssh -T git@github.com)"

# Main command
sudo DOCKER_BUILDKIT=1 docker build --progress=plain --ssh default="${SSH_AUTH_SOCK}" .

# Clean up
eval "$(ssh-agent -k)"
