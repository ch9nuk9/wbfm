# syntax=docker/dockerfile:1

# From: https://medium.com/@tonistiigi/build-secrets-and-ssh-forwarding-in-docker-18-09-ae8161d066
FROM alpine
# install ssh client and git
RUN apk add --no-cache openssh-client git
# download public key for github.com
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh echo "Available ssh keys: $(ssh-add -L)"
# clone our private repository
RUN --mount=type=ssh git clone git@github.com:Zimmer-lab/dlc_for_wbfm.git dlc_for_wbfm

# Run tests of the installation
