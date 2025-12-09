* /home/wl/下载/FlashOCC/doc/nuscenes_det.md 
* cd /home/wl/下载/FlashOCC/mmdetection3d

    python tools/create_data.py nuscenes \
        --root-path ./data/nuscenes \
        --out-dir ./data/nuscenes \
        --extra-tag nuscenes \
        --version v1.0-mini


将mmdetection3d 目录下的data目录也放在了/home/wl/下载/FlashOCC/data。软连接。本身这个data目录就有好几层软连接了，
python tools/test.py \
    projects/configs/flashocc/flashocc-r50.py \
    ckpts/flashocc-r50-256x704.pth \
    --eval map

cd /home/wl/下载/FlashOCC

# 运行测试并保存结果
bash tools/dist_test.sh projects/configs/flashocc/flashocc-r50.py ckpts/flashocc-r50-256x704.pth １ --eval map --eval-options show_dir=work_dirs/flashocc_r50/results
# step 2. visualization
python tools/analysis_tools/vis_occ.py work_dirs/flashocc_r50/results/ --root_path . --save_path ./vis


# 2025.12.9
conda activate FlashOcc
cd /home/wl/下载/FlashOCC
python tools/create_data_bevdet.py  # ✅ Correct