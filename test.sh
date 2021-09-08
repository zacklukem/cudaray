#!/bin/bash

make -s -j 32 && \
echo "Running..." && \
time ./build/cudaray && \
echo "Done..." && \
cp "./test.png" "$HOME/.www" && \
chmod a+r "$HOME/.www/test.png" && \
convert -enhance -resize 200% test.png test2.png && \
imgcat test2.png
