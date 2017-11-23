VER=1

CUDA_VISIBLE_DEVICES=1 python crnn_test.py \
                        --cuda \
                        --experiment outputs/CRNN_v$VER \
                        --workers 0 \
                        --niter 30 \
                        --displayInterval 1 \
                        --lr 0.001 \
                        --crnn outputs/CRNN_v1/netCRNN_0_43000.pth \
                        --font_path fonts/kaiti.ttf,fonts/songti.TTF,fonts/songticu.ttf,fonts/xiyuan.ttf,fonts/yahei.TTF,fonts/zhengheiti.ttf \
                        --fontsize 20-48 \
                        --val_list data/val_text.txt \
                        --random_sample \
                        --imgH 32 \
                        --imgW 100 \
                        --keep_ratio \
                        --ngpu 1