#!/bin/bash

make && \
echo "Running..." && \
./build/cudaray && \
echo "Done..." && \
cp "./test.png" "$HOME/.www" && \
chmod a+r "$HOME/.www/test.png"
