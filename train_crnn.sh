VER=0314
CUDA_VISIBLE_DEVICES=1 python -u  crnn_train.py \
                        --cuda \
                        --experiment outputs/CRNN_v$VER \
                        --batchSize 64 \
                        --workers 4 \
                        --niter 200 \
                        --nh 256 \
                        --valInterval 500 \
                        --displayInterval 1 \
                        --saveInterval 8000 \
                        --lr 0.00005 \
                        --trainroot ../dataset/crop_plate_all_lmdb/ \
                        --valroot ../dataset/crop_plate_lingang_lmdb/ \
                        --random_sample \
                        --imgH 32 \
                        --imgW 100 \
                        --keep_ratio \
                        --ngpu 1 
