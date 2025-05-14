#!/bin/bash

# Get data
pushd data
    python get_uci.py
    python get_md22.py
popd