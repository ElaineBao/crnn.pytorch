python crnn_test.py \
    --nh 256 \
    --workers 0 \
    --crnn outputs/CRNN_v0314/netCRNN_28_8000.pth  \
    --valroot ../dataset/crop_plate_lingang_lmdb/ \
    --imgH 32 \
    --imgW 100
