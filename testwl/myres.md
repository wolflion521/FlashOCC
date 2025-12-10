# 2025.12.10
## 结果001
    Starting Evaluation...
    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6019/6019 [00:24<00:00, 245.56it/s]
    ===> per class IoU of 6019 samples:
    ===> others - IoU = 6.34
    ===> barrier - IoU = 38.36
    ===> bicycle - IoU = 11.13
    ===> bus - IoU = 39.14
    ===> car - IoU = 44.19
    ===> construction_vehicle - IoU = 14.26
    ===> motorcycle - IoU = 14.23
    ===> pedestrian - IoU = 16.18
    ===> traffic_cone - IoU = 15.12
    ===> trailer - IoU = 27.59
    ===> truck - IoU = 31.43
    ===> driveable_surface - IoU = 78.4
    ===> other_flat - IoU = 37.99
    ===> sidewalk - IoU = 48.32
    ===> terrain - IoU = 52.06
    ===> manmade - IoU = 38.02
    ===> vegetation - IoU = 32.26
    ===> mIoU of 6019 samples: 32.06
    {'mIoU': array([0.063, 0.384, 0.111, 0.391, 0.442, 0.143, 0.142, 0.162, 0.151,
        0.276, 0.314, 0.784, 0.38 , 0.483, 0.521, 0.38 , 0.323, 0.883])}


python tools/test.py     projects/configs/flashocc/flashocc-stbase-4d-stereo-512x1408_4x4_2e-4.py     ckpts/flashocc-stbase-4d-stereo-512x1408.pth     --eval map