python -u test_erfnet.py CULane ERFNet train test_img \
                          --lr 0.01 \
                          --gpus 0 \
                          --resume trained/ERFNet_trained.tar \
                          --img_height 208 \
                          --img_width 976 \
                          -j 10 \
                          -b 1
