VER=1

CUDA_VISIBLE_DEVICES=0 python crnn_train.py \
                        --cuda \
                        --experiment outputs/CRNN_v$VER \
                        --workers 1 \
                        --niter 10 \
                        --displayInterval 1 \
                        --saveInterval 1000 \
                        --lr 0.00005 \
                        --crnn models/netCRNN63.pth \
                        --font_path fonts/kaiti.ttf,fonts/songti.TTF,fonts/songticu.ttf,fonts/xiyuan.ttf,fonts/yahei.TTF,fonts/zhengheiti.ttf \
                        --fontsize 20-48 \
                        --train_list data/train_text.txt \
                        --random_sample \
                        --imgH 32 \
                        --imgW 100 \
                        --keep_ratio \
                        --ngpu 1