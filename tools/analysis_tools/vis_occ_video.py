"""
简洁高效的 FlashOCC 可视化工具
生成每个scene的mp4视频，包含前3个摄像头图片和occupancy 3D可视化
"""
import os
import argparse
import pickle
import numpy as np
import cv2
import torch
import open3d as o3d
from tqdm import tqdm

# Occupancy 配置
FREE_LABEL = 17
VOXEL_SIZE = [0.4, 0.4, 0.4]
POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]

# 18类occupancy颜色映射
COLORMAP = np.array([
    [0, 0, 0],           # 0 undefined
    [112, 128, 144],     # 1 barrier
    [220, 20, 60],       # 2 bicycle
    [255, 127, 80],      # 3 bus
    [255, 158, 0],       # 4 car
    [233, 150, 70],      # 5 construction vehicle
    [255, 61, 99],       # 6 motorcycle
    [0, 0, 230],         # 7 pedestrian
    [47, 79, 79],        # 8 traffic cone
    [255, 140, 0],       # 9 trailer
    [255, 99, 71],       # 10 truck
    [0, 207, 191],       # 11 driveable surface
    [175, 0, 75],        # 12 other flat
    [75, 0, 75],         # 13 sidewalk
    [112, 180, 60],      # 14 terrain
    [222, 184, 135],     # 15 manmade
    [0, 175, 0],         # 16 vegetation
], dtype=np.uint8)


def voxel_to_points(voxel_label, mask):
    """将体素标签转换为3D点云"""
    valid_mask = np.logical_and(voxel_label != FREE_LABEL, mask)
    coords = np.argwhere(valid_mask)  # (N, 3): (x_idx, y_idx, z_idx)
    
    # 转换索引到世界坐标
    points = coords.astype(np.float32)
    points[:, 0] = points[:, 0] * VOXEL_SIZE[0] + POINT_CLOUD_RANGE[0]
    points[:, 1] = points[:, 1] * VOXEL_SIZE[1] + POINT_CLOUD_RANGE[1]
    points[:, 2] = points[:, 2] * VOXEL_SIZE[2] + POINT_CLOUD_RANGE[2]
    
    labels = voxel_label[valid_mask]
    colors = COLORMAP[labels % len(COLORMAP)] / 255.0
    
    return points, colors


def create_voxel_visualization(points, colors, img_width=1200, img_height=800):
    """创建occupancy的3D可视化并渲染为图片"""
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建体素网格以获得更好的可视化效果
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.4)
    
    # 创建离线渲染器
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_width, height=img_height, visible=False)
    vis.add_geometry(voxel_grid)
    
    # 设置相机视角（车顶3.5m高度，向前俯视10度）
    ctr = vis.get_view_control()
    # 相机位置在车顶正上方3.5m处 (ego车在原点)
    # 向前看（+Y方向），向下俯视10度
    import math
    pitch = math.radians(10)  # 俯视10度
    
    ctr.set_lookat([0, 20, 0])          # 看向前方20m处的地面
    ctr.set_front([0, math.cos(pitch), -math.sin(pitch)])  # 向前+向下
    ctr.set_up([0, math.sin(pitch), math.cos(pitch)])      # 上方向
    ctr.set_zoom(0.35)                  # 调整缩放以看到更多范围
    
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # 白色背景
    opt.point_size = 2.0
    
    # 渲染并捕获图像
    vis.poll_events()
    vis.update_renderer()
    
    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    
    # 转换为OpenCV格式
    img = (np.asarray(img) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img


def create_frame(cam_imgs, occ_img, cam_height=400):
    """
    创建单帧可视化图像
    Args:
        cam_imgs: list of 3 camera images [CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT]
        occ_img: occupancy visualization image
        cam_height: 摄像头图片的目标高度
    Returns:
        combined_frame: (H, W, 3) BGR image
    """
    # 调整3个摄像头图片到统一高度
    resized_cams = []
    for img in cam_imgs:
        h, w = img.shape[:2]
        new_w = int(w * cam_height / h)
        resized = cv2.resize(img, (new_w, cam_height))
        resized_cams.append(resized)
    
    # 水平拼接3个摄像头图片
    cam_row = np.hstack(resized_cams)
    
    # 调整occupancy图片宽度与摄像头行对齐
    cam_row_h, cam_row_w = cam_row.shape[:2]
    occ_h, occ_w = occ_img.shape[:2]
    
    # 保持occupancy图片比例，调整宽度
    new_occ_h = int(occ_h * cam_row_w / occ_w)
    occ_resized = cv2.resize(occ_img, (cam_row_w, new_occ_h))
    
    # 垂直拼接：上方摄像头，下方occupancy
    frame = np.vstack([cam_row, occ_resized])
    
    return frame


def visualize_scene(scene_data, pred_dir, gt_dir, output_path, fps=5):
    """
    为单个scene生成可视化视频
    Args:
        scene_data: list of sample info dicts for this scene
        pred_dir: 预测结果目录
        gt_dir: ground truth 目录
        output_path: 输出视频路径
        fps: 视频帧率
    """
    frames = []
    
    print(f"  Processing {len(scene_data)} samples...")
    for sample_info in tqdm(scene_data, desc="  Rendering"):
        scene_name = sample_info['scene_name']
        token = sample_info['token']
        
        # 加载预测的occupancy
        pred_path = os.path.join(pred_dir, scene_name, token, 'pred.npz')
        if not os.path.exists(pred_path):
            print(f"  Warning: {pred_path} not found, skipping")
            continue
        
        pred_occ = np.load(pred_path)['pred']
        
        # 加载GT mask
        gt_path = sample_info['occ_path']  # 格式: './data/nuscenes/gts/scene-XXX/token'
        # 去掉开头的 './' 和 gt_dir 中可能重复的 'data/nuscenes'
        if gt_path.startswith('./'):
            gt_path = gt_path[2:]  # 去掉 './'
        # gt_path 现在是 'data/nuscenes/gts/...'
        # 直接从 gt_dir (项目根目录) 拼接
        gt_full_path = os.path.join(gt_dir, gt_path, 'labels.npz')
        
        gt_data = np.load(gt_full_path)
        camera_mask = gt_data['mask_camera']
        
        # 加载前3个摄像头图片
        cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
        cam_imgs = []
        for cam_name in cam_names:
            img_path = sample_info['cams'][cam_name]['data_path']
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Warning: {img_path} not found")
                img = np.zeros((900, 1600, 3), dtype=np.uint8)
            cam_imgs.append(img)
        
        # 生成occupancy 3D可视化
        points, colors = voxel_to_points(pred_occ, camera_mask)
        
        if len(points) == 0:
            # 如果没有点，创建空白图片
            occ_img = np.ones((800, 1200, 3), dtype=np.uint8) * 255
        else:
            occ_img = create_voxel_visualization(points, colors, 1200, 800)
        
        # 组合成单帧
        frame = create_frame(cam_imgs, occ_img, cam_height=400)
        frames.append(frame)
    
    if len(frames) == 0:
        print("  No frames generated, skipping video creation")
        return
    
    # 写入视频
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in frames:
        video_writer.write(frame)
    
    video_writer.release()
    print(f"  ✓ Video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate visualization videos for FlashOCC')
    parser.add_argument('pred_dir', help='Prediction results directory (e.g., work_dirs/flashocc_r50/results/)')
    parser.add_argument('--data-root', default='.', help='Project root path')
    parser.add_argument('--info-file', default='data/nuscenes/bevdetv2-nuscenes_infos_val.pkl',
                        help='Info file path relative to data-root')
    parser.add_argument('--output-dir', default='./vis_videos', help='Output directory for videos')
    parser.add_argument('--fps', type=int, default=5, help='Video frame rate')
    parser.add_argument('--max-scenes', type=int, default=None, help='Max number of scenes to process')
    
    args = parser.parse_args()
    
    # 加载数据集信息
    info_path = os.path.join(args.data_root, args.info_file)
    print(f"Loading dataset info from {info_path}...")
    with open(info_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # 按scene分组
    scene_dict = {}
    for sample_info in dataset['infos']:
        scene_name = sample_info['scene_name']
        if scene_name not in scene_dict:
            scene_dict[scene_name] = []
        scene_dict[scene_name].append(sample_info)
    
    print(f"Found {len(scene_dict)} scenes with {len(dataset['infos'])} total samples")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 为每个scene生成视频
    scene_names = sorted(scene_dict.keys())
    if args.max_scenes:
        scene_names = scene_names[:args.max_scenes]
    
    for idx, scene_name in enumerate(scene_names):
        print(f"\n[{idx+1}/{len(scene_names)}] Processing scene: {scene_name}")
        output_path = os.path.join(args.output_dir, f'{scene_name}.mp4')
        
        visualize_scene(
            scene_data=scene_dict[scene_name],
            pred_dir=args.pred_dir,
            gt_dir=args.data_root,  # 直接使用项目根目录
            output_path=output_path,
            fps=args.fps
        )
    
    print(f"\n✓ All done! Videos saved to {args.output_dir}")


if __name__ == '__main__':
    main()
