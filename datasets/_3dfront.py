import os, sys, glob, torch
import numpy as np
import random
[sys.path.append(i) for i in ['.', '..']]
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from lib.benchmark_utils import to_o3d_pcd, to_tsfm, get_correspondences


class _3DFront(Dataset):

    def __init__(self, config, split, data_augmentation=True):
        super(_3DFront, self).__init__()

        assert split in ['train', 'val', 'test']

        if 'overfit' in config.exp_dir:
            d_slice = config.batch_size
        else:
            d_slice = None
        self.config = config
        self.infos = self.read_entries(split, config.data_root, d_slice=d_slice)

        self.base_dir = config.data_root
        self.data_augmentation = data_augmentation
        

        self.rot_factor = 1.
        self.augment_noise = config.augment_noise
        self.max_points = config.max_points
        print('max_points', self.max_points)
        self.overlap_radius = 0.0375

    def read_entries(self, split, data_root, d_slice=None, shuffle=True):
        if split == 'train':
            file_s = [10000,2800,6000,2000]
            entries = self._build_data_list(os.path.join(data_root, 'pro25/high'), file_s[0])
            entries.extend(self._build_data_list(os.path.join(data_root, 'pro25/low'), file_s[1]))
            entries.extend(self._build_data_list(os.path.join(data_root, 'pro40/high'), file_s[2]))
            entries.extend(self._build_data_list(os.path.join(data_root, 'pro40/low'), file_s[3]))
            if d_slice:
                entries = entries[:d_slice]
        elif split == 'val':
            file_s = [100,40,70,20]
            entries = self._build_data_list(os.path.join(data_root, 'pro25/high'), file_s[0], val = True)
            entries.extend(self._build_data_list(os.path.join(data_root, 'pro25/low'), file_s[1],val = True))
            entries.extend(self._build_data_list(os.path.join(data_root, 'pro40/high'), file_s[2], val = True))
            entries.extend(self._build_data_list(os.path.join(data_root, 'pro40/low'), file_s[3], val = True))
            if d_slice:
                entries = entries[:d_slice]
        else:
            entries = self._build_data_list(os.path.join(data_root, self.config.test_name), 1000)
            if d_slice:
                entries = entries[:d_slice]
        
        return entries
    
    def _build_data_list(self,dir,file_number=2000, val=False):
        #[10000,2800,6000,2000]
        #[1000,400,700,200]
        data_list = []
        total = 0
        scene_ids = os.listdir(dir)
        if val:
            scene_ids = sorted(scene_ids, reverse=True)
        for scene_id in scene_ids:
            scene_path = os.path.join(dir, scene_id)
            if os.path.isdir(scene_path):
                data_list.append(scene_path)
                total += 1
                if total >= file_number:
                    break
        return data_list

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, item, debug=False):
        entry = self.infos[item]
        src_path = os.path.join(entry, 'src.npy')
        if self.config.wo_anim:
            ref_path = os.path.join(entry, 'ref_wo_anim.npy')
        else:
            ref_path = os.path.join(entry, 'ref.npy')
        #ref_path = os.path.join(entry, 'ref_wo_anim.npy')
        gt_path = os.path.join(entry, 'relative_transform.npy')

        src_pcd = np.load(src_path)
        tgt_pcd = np.load(ref_path)
        gt_tsfm = np.load(gt_path)
        rot = gt_tsfm[:3, :3]
        trans = gt_tsfm[:3, 3].reshape(3,1)
        gt_cov = None  # 如果没有提供真值协方差，则设为 None

        # 如果点云点数过多，则进行下采样
        if src_pcd.shape[0] > self.max_points:
            idx = np.random.permutation(src_pcd.shape[0])[:self.max_points]
            src_pcd = src_pcd[idx]
        if tgt_pcd.shape[0] > self.max_points:
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
            tgt_pcd = tgt_pcd[idx]

        
        # 添加高斯噪声
        if self.data_augmentation:
            # # 旋转点云
            # euler_ab = np.random.rand(3) * np.pi * 2 / self.rot_factor  # zyx 角度
            # rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            # if np.random.rand(1)[0] > 0.5:
            #     src_pcd = np.matmul(rot_ab, src_pcd.T).T
            #     rot = np.matmul(rot, rot_ab.T)
            # else:
            #     tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T
            #     rot = np.matmul(rot_ab, rot)
            #     trans = np.matmul(rot_ab, trans)

            src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise

        # 获取点对对应关系
        tsfm = to_tsfm(rot, trans)
        correspondences = get_correspondences(to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd), tsfm, self.overlap_radius)

        # # 在调试模式下，再次可视化
        # if debug:
        #     import mayavi.mlab as mlab
        #     c_red = (224. / 255., 0 / 255., 125 / 255.)
        #     c_blue = (0. / 255., 0. / 255., 255. / 255.)
        #     scale_factor = 0.02
        #     mlab.points3d(src_pcd[:, 0], src_pcd[:, 1], src_pcd[:, 2], scale_factor=scale_factor, color=c_red)
        #     mlab.points3d(tgt_pcd[:, 0], tgt_pcd[:, 1], tgt_pcd[:, 2], scale_factor=scale_factor, color=c_blue)
        #     mlab.show()

        # if trans.ndim == 1:
        #     trans = trans[:, None]

        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)
        return src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, rot, trans, gt_cov
