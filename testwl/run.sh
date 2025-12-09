# ========== 阶段1: 解压前5个blob (01-05) ==========
cd /home/wl/下载/data/nuscenes
BLOBS_DIR="/home/wl/下载/temp/train"

# 先解压meta
tar -xzf "${BLOBS_DIR}/v1.0-trainval_meta.tgz" && \
rm "${BLOBS_DIR}/v1.0-trainval_meta.tgz"

# 解压blob 01-05
for i in {01..05}; do
    echo "🔄 解压 v1.0-trainval${i}_blobs.tgz"
    tar -xzf "${BLOBS_DIR}/v1.0-trainval${i}_blobs.tgz" \
        --checkpoint=.10000 \
        --checkpoint-action=echo="%u files extracted"
    
    if [ $? -eq 0 ]; then
        rm "${BLOBS_DIR}/v1.0-trainval${i}_blobs.tgz"
        echo "✅ blob ${i} 完成,已删除压缩包"
        df -h /home/wl/下载 | grep nvme | awk '{print "📊 可用空间:", $4}'
    fi
done

echo "✅ 阶段1完成,已释放约140GB空间 (blob 01-05)"

# ========== 阶段2: 解压后5个blob (06-10) ==========
for i in {06..10}; do
    echo "🔄 解压 v1.0-trainval${i}_blobs.tgz"
    tar -xzf "${BLOBS_DIR}/v1.0-trainval${i}_blobs.tgz" \
        --checkpoint=.10000 \
        --checkpoint-action=echo="%u files extracted"
    
    if [ $? -eq 0 ]; then
        rm "${BLOBS_DIR}/v1.0-trainval${i}_blobs.tgz"
        echo "✅ blob ${i} 完成,已删除压缩包"
        df -h /home/wl/下载 | grep nvme | awk '{print "📊 可用空间:", $4}'
    fi
done

echo "🎉 所有训练数据解压完成!"
