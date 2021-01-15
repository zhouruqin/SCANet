import torch
import numpy as np
import torchvision
import src.quaternion as Q  # works with (w, x, y, z) quaternions
import kornia.geometry.conversions as C  # works with (x, y, z, w) quaternions
import kornia.geometry.linalg as L
from sklearn.metrics import r2_score
from scipy.spatial.transform import Rotation
import src.pctransforms  as Transformation

def deg_to_rad(deg):
    return np.pi / 180 * deg


def rad_to_deg(rad):
    return 180 / np.pi * rad


 
def rotationMatrixToAngles(R):
 
    sy = torch.sqrt(R[:, 0, 0] * R[:,0, 0] + R[:,1, 0] * R[:,1, 0])  #矩阵元素下标都从0开始（对应公式中是sqrt(r11*r11+r21*r21)），sy=sqrt(cosβ*cosβ)
 
    #singular = sy < 1e-6   # 判断β是否为正负90°
 
    #if not singular:   #β不是正负90°
    x = torch.atan2(R[:,2, 1], R[:,2, 2])
    y = torch.atan2(-R[:,2, 0], sy)
    z = torch.atan2(R[:,1, 0], R[:,0, 0])
    #else:              #β是正负90°
    #    x = torch.atan2(-R[:,1, 2], R[:,1, 1])
    #    y = torch.atan2(-R[:,2, 0], sy)   #当z=0时，此公式也OK，上面图片中的公式也是OK的
    #    z = 0
    return torch.cat([x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)], dim=1)


class QuaternionTransform:
    def __init__(self, vec: torch.Tensor, inverse: bool = False):
        # inversion: first apply translation
        self._inversion = torch.tensor([inverse])
        # a B x 7 vector of 4 quaternions and 3 translation parameters
        self.vec = vec.view([-1, 7])

    # Dict constructor
    @staticmethod
    def from_dict(d, device):
        return QuaternionTransform(d["vec"].to(device), d["inversion"][0].item())

    # Inverse Constructor
    def inverse(self):
        quat = self.quat()
        trans = self.trans()
        quat = Q.qinv(quat)
        trans = -trans

        vec = torch.cat([quat, trans], dim=1)
        return QuaternionTransform(vec, inverse=(not self.inversion()))

    def as_dict(self):
        return {"inversion": self._inversion, "vec": self.vec}
    
    def angle(self):
        return C.quaternion_to_angle_axis(self.quat())#

    def quat(self):
        return self.vec[:, 0:4]
    
    def rotate(self):
        return C.quaternion_to_rotation_matrix(self.wxyz_to_xyzw(self.quat())) #

    def trans(self):
        return self.vec[:, 4:]

    def inversion(self):
        # To handle dataloader batching of samples,
        # we take the first item's 'inversion' as the inversion set for the entire batch.
        return self._inversion[0].item()

    @staticmethod
    def wxyz_to_xyzw(q):
        q = q[..., [1, 2, 3, 0]]
        return q

    @staticmethod
    def xyzw_to_wxyz(q):
        q = q[..., [3, 0, 1, 2]]
        return q

    def dcm2euler( self, mats: np.ndarray, seq: str = 'zyx', degrees: bool = True):
        """Converts rotation matrix to euler angles

            Args:
                mats: (B, 3, 3) containing the B rotation matricecs
                seq: Sequence of euler rotations (default: 'zyx')
                degrees (bool): If true (default), will return in degrees instead of radians

            Returns:
        """

        eulers = []
        for i in range(mats.shape[0]):
            r = Rotation.from_dcm(mats[i])
            eulers.append(r.as_euler(seq, degrees=degrees))
        return np.stack(eulers)

    def compute_errors(self, other):
        q1 = self.quat()
        q2 = other.quat()
        angle_1 = self.angle()
        angle_2 = other.angle()
        #angle_1 = Q.qeuler(q1,  order ="xyz")
        #angle_2 = Q.qeuler(q2,  order ="xyz")
        #C.quaternion_to_angle_axis(q1)
        #R1 = C.quaternion_to_rotation_matrix(self.wxyz_to_xyzw(q1))
        #R2 = C.quaternion_to_rotation_matrix(self.wxyz_to_xyzw(q2))
        #R2inv = R2.transpose(1, 2)
        #R1_R2inv = torch.bmm(R1, R2inv)

        #r_ab_mse = ((R1_R2inv[:, 0,0] + R1_R2inv[:, 1,1] + R1_R2inv[:, 2,2])**2
        r_ab_mse = torch.sqrt((angle_1[:, 0]-angle_2[:, 0])**2 + 
            (angle_1[:, 1]-angle_2[:, 1])**2 +
            (angle_1[:, 2]-angle_2[:, 2])**2)
        #print(angle_1)
        #print(angle_2)
        #torch.sqrt((q1[:, 0] - q2[:, 0])** 2 +
        #    (q1[:, 1] - q2[:, 1])** 2 +
        #    (q1[:, 2] - q2[:, 2])** 2 +
        #    (q1[:, 3] - q2[:, 3])** 2 )#( q1-q2)** 2
        #
        #(angle_1-angle_2)**2
        #r_ab_mse = (2 * (torch.sum(q1 * q2, dim=1)) ** 2 - 1)
        #batch = R1_R2inv.shape[0]
        #I = torch.eye(3).unsqueeze(0).expand([batch, -1, -1]).to(R1_R2inv)
        #r_ab_mse = (R1_R2inv - I) ** 2
        #r_ab_mse = torch.mean(norm_err)#(self.trans()- other.trans())** 2
       
        t_ab_mse =  torch.sqrt((self.trans()[:, 0] - other.trans()[:, 0])** 2 +
            (self.trans()[:, 1] - other.trans()[:, 1])** 2 +
            (self.trans()[:, 2] - other.trans()[:, 2])** 2)
        
        return r_ab_mse,  t_ab_mse

    def apply_transform(self, p: torch.Tensor):
        ndim = p.dim()

        #print('ndim',ndim)
        if ndim == 2:
            N, _ = p.shape
            assert self.vec.shape[0] == 1
            # repeat transformation vector for each point in shape
            quat = self.quat().expand([N, -1])
            trans = self.trans().expand([N, -1]) 
            p_transformed =  Q.qrot(quat, p) + trans#
        elif ndim == 3:
            B, N, _ = p.shape
            quat = self.quat().unsqueeze(1).expand([-1, N, -1]).contiguous()
            trans = self.trans().unsqueeze(1).expand([-1, N, -1]).contiguous()
            p_transformed =  Q.qrot(quat, p) + trans#

        return p_transformed


def create_random_transform(dtype, max_rotation_deg, max_translation):
    max_rotation = deg_to_rad(max_rotation_deg)
    rot = np.random.uniform(0, max_rotation, [1, 3])
    trans = np.random.uniform(-max_translation, max_translation, [1, 3])
    quat = Q.euler_to_quaternion(rot, "xyz")

    vec = np.concatenate([quat, trans], axis=1)
    vec = torch.tensor(vec, dtype=dtype)
    return QuaternionTransform(vec)


class QuaternionFixedDataset(torch.utils.data.Dataset):
    def __init__(self, args, data, repeat=1, seed=0):
        super().__init__()
        self.data = data
        self.include_shapes = data.include_shapes
        self.len_data = len(data)
        self.len_set = len(data) * repeat

        np.random.seed(seed)
        #生成gt变换矩阵
        self.transforms = [
            create_random_transform(torch.float32, 45, 0.5) for _ in range(self.len_set)
        ]
        #数据增广
        self.data_enhence = torchvision.transforms.Compose( [
                            Transformation.PointcloudRotatePerturbation(),
                            ])
        #数据模拟S
        if args.noise_type == 'clean':
            self.transforms_1 = torchvision.transforms.Compose( [ 
                Transformation.OnUnitCube(),
            ])
            
        elif args.noise_type == 'crop':
            self.transforms_1 =  torchvision.transforms.Compose( [
                Transformation.OnUnitCube(),
                Transformation.PointcloudCrop(),
                Transformation.PointcloudJitter(),
            ])
            
        elif args.noise_type == 'jitter':
            self.transforms_1 =  torchvision.transforms.Compose( [
                Transformation.OnUnitCube(),
                Transformation.PointcloudJitter(),
            ])
            
        elif args.noise_type == 'part':
            self.transforms_1 =   torchvision.transforms.Compose([
                Transformation.OnUnitCube(),
                Transformation.PointcloudCrop(),
            ])
            

        self.shuffle =  Transformation.ShufflePoints()  
      

    def __len__(self):
        return self.len_set

    def __getitem__(self, index):

        if self.include_shapes:
            points, _, shape = self.data[index % self.len_data]
        else:
            points, _ = self.data[index % self.len_data]
        
        #splite source and target
        points = self.data_enhence(points)

        p_source = points.detach()
        p_target = points.detach()
        
        p_source = self.transforms_1(p_source)    # add jitter, crop, part
        p_target = self.transforms_1(p_target)
        #add (R,t) to source 
        gt = self.transforms[index]
        p_target = gt.apply_transform(p_target)   #target = (R,T)* source
        p_points = gt.apply_transform(points)

        igt = gt.as_dict()  # p0 ---> p1
        p_source_1 = self.shuffle(p_source)
        p_target_1 = self.shuffle(p_target)
        
        if self.include_shapes:
            return p_source_1, p_target_1, p_points, igt, shape

        return p_source_1, p_target_1, p_points, igt


if __name__ == "__main__":
    toy = np.array([[[1.0, 1.0, 1.0], [2, 2, 2]], [[0.0, 1.0, 0.0], [0, 2, 0]]])
    toy = torch.tensor(toy, dtype=torch.float32)
