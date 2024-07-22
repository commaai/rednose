#!/bin/bash
brew bundle --file=- <<-EOS
brew "eigen"
cask "gcc-arm-embedded"
brew "gcc@13"
EOS
