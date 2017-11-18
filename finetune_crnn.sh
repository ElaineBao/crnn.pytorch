VER=1

CUDA_VISIBLE_DEVICES=0 python crnn_train.py \
                        --cuda \
                        --experiment outputs/CRNN_v$VER \
                        --workers 4 \
                        --niter 30 \
                        --step 10 \
                        --displayInterval 1 \
                        --lr 0.001 \
                        --crnn models/netCRNN63.pth \
                        --font_path fonts/kaiti.ttf,fonts/songti.TTF,fonts/songticu.ttf,fonts/xiyuan.ttf,fonts/yahei.TTF,fonts/zhengheiti.ttf \
                        --fontsize 32-48 \
                        --val_list data/train_text.txt \
                        --train_list data/val_text.txt \
                        --imgH 32 \
                        --imgW 130 \
                        --keep_ratio \
                        --ngpu 2