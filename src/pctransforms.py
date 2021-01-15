from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import numpy as np


def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()

def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere

    Source: https://gist.github.com/andrewbolster/10274979

    Args:
        num: Number of vectors to sample (or None if single)

    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)

    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    #xyz =  torch.from_numpy( np.stack((x, y, z), axis=-1))
    return np.stack((x, y, z), axis=-1)


class PointcloudScale(object):
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points


class PointcloudRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def __call__(self, points):
        angles = self._get_angles()
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = (
            points.new(points.size(0), 3)
            .normal_(mean=0.0, std=self.std)
            .clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points


class PointcloudCrop(object):
    def __init__(self, p_keep= 0.7):
        self.p_keep = np.array(p_keep, dtype=np.float32)

    def __call__(self, points):
       
        rand_xyz = uniform_2_sphere()
        pts = points.numpy()
        centroid = np.mean(pts[:, :3], axis=0)
        points_centered = pts[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if self.p_keep ==  0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - self.p_keep) * 100)
        #print("Crop generated!")
        return torch.Tensor(pts[mask, :3])

class PointcloudTranslate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, points):
        translation = np.random.uniform(-self.translate_range, self.translate_range)
        points[:, 0:3] += translation
        return points


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.1):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points):
        pc = points.numpy()

        dropout_ratio =  self.max_dropout_ratio  # 0~0.875np.random.random() *
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            pc[drop_idx] = pc[0]  # set to the first point

        return torch.from_numpy(pc).float()


class OnUnitCube(object):
    def __init__(self):
        pass

    def method1(self, tensor):
        m = tensor.mean(dim=0, keepdim=True)  # [N, D] -> [1, D]
        v = tensor - m
        s = torch.max(v.abs())
        v = v / s * 0.5
        return v

    def method2(self, tensor):
        xyz_min = torch.max(tensor, dim=0)[0]
        xyz_max = torch.min(tensor, dim=0)[0]
        xyz_move = xyz_min+(xyz_max-xyz_min)/2
        tensor[:,0:3] = tensor[:,0:3]-xyz_move
        #scale
        scale = torch.max(tensor[:,0:3])
        tensor[:,0:3] = tensor[:,0:3]/scale
        return tensor
    '''
        c = torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0]  # [N, D] -> [D]
        s = torch.max(c)  # -> scalar
        v = tensor / s
        return v - v.mean(dim=0, keepdim=True)
    '''

    def __call__(self, tensor):
        # return self.method1(tensor)
        return self.method2(tensor)


class ShufflePoints(object):
    """Shuffles the order of the points"""
    def __call__(self, points):
        return torch.Tensor(np.random.permutation(points))