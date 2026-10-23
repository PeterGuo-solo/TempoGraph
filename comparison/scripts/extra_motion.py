"""Coordinate conversion only; no smoothing, translation removal or motion editing."""
import math
import numpy as np

def window_layout(frames):
    if frames < 1: raise ValueError('frames must be positive')
    count=max(1,math.ceil((frames-120)/60)+1)
    return count,120+60*(count-1)

def canonical_motion(poses,translations,frames,method):
    import torch
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
    poses=np.asarray(poses,dtype=np.float32)
    translations=np.asarray(translations,dtype=np.float32)
    if poses.ndim!=2 or poses.shape[1] not in (66,72,156):
        raise ValueError('Unexpected axis-angle shape: '+str(poses.shape))
    if len(poses)<frames or translations.shape!=(len(poses),3):
        raise ValueError('Invalid motion length or translation shape')
    if not np.isfinite(poses).all() or not np.isfinite(translations).all():
        raise ValueError('Nonfinite model output')
    r=axis_angle_to_matrix(torch.from_numpy(poses[:frames].reshape(frames,-1,3)[:,:22].copy()))
    root=torch.from_numpy(translations[:frames].copy())
    if method=='flowerdance':
        # Inverse of the official training loader's +90 degree X world rotation.
        inverse=torch.tensor([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])
        r[:,0]=inverse@r[:,0]
        root=root@inverse.T
    elif method!='finenet':raise ValueError(method)
    return matrix_to_rotation_6d(r).numpy(),root.numpy()
