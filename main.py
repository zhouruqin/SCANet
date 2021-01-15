import argparse
import logging
import os
import sys

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from data.modelnet_loader_torch import ModelNetCls
from models import pcrnet
from src import ChamferDistance, FPSSampler, RandomSampler, SampleNet
from src import sputils
from src.pctransforms import OnUnitCube, PointcloudToTensor, PointcloudCrop, PointcloudJitter, PointcloudRandomInputDropout
from src.qdataset import QuaternionFixedDataset, QuaternionTransform, rad_to_deg, create_random_transform
from sklearn.metrics import r2_score
from scipy.spatial.transform import Rotation
import kornia.geometry.conversions as C

torch.manual_seed(0)

# addpath('../')
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

# dump to GLOBALS dictionary
GLOBALS = None


def append_to_GLOBALS(key, value):
    try:
        GLOBALS[key].append(value)
    except KeyError:
        GLOBALS[key] = []
        GLOBALS[key].append(value)


# fmt: off
def options(argv=None, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--outfile', required=True, type=str,
                        metavar='BASENAME', help='output filename (prefix)')  # the result: ${BASENAME}_model_best.pth
    parser.add_argument('--datafolder', default= 'modelnet40_ply_hdf5_2048', type=str, help='dataset folder')


    # For testing
    parser.add_argument('--test', action='store_true',
                        help='Perform testing routine. Otherwise, the script will train.')
    parser.add_argument('--apply', action='store_true',
                        help='Perform testing routine. Otherwise, the script will train.')

    # Default pointnet behavior is 'fixed'.
    # Loading options:
    #   --transfer-from: load a pretrained PCRNET model.
    #   --resume: load an ongoing training SP-PCRNET model.
    #   --pretrained: load a pretrained SP-PCRNET model (like resume, but reset starting epoch)

    parser.add_argument('--loss-type', default=0, choices=[0, 1], type=int,
                        metavar='TYPE', help='Supervised (0) or Unsupervised (1)')
    parser.add_argument('--sampler', required=True, choices=['fps', 'samplenet', 'random', 'none'], type=str,
                        help='Sampling method.')

    parser.add_argument('--transfer-from', type=str,
                        metavar='PATH', help='path to trained pcrnet')
    parser.add_argument('--train-pcrnet', action='store_true',
                        help='Allow PCRNet training.')
    parser.add_argument('--train-samplenet', action='store_true',
                        help='Allow SampleNet training.')

    parser.add_argument('--num-sampled-clouds', choices=[1, 2], type=int, default=2,
                        help='Number of point clouds to sample (Source / Source + Template)')

    # settings for on training
    parser.add_argument('--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--epochs', default=400, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD', 'RMSProp'],
                        metavar='METHOD', help='name of an optimizer (default: Adam)')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')
    parser.add_argument('--noise_type', default='crop', choices=['clean', 'jitter', 'crop', 'part'],
                        help='Types of perturbation to consider')

    args = parser.parse_args(argv)
    return args



def main(args, dbg=False):
    global GLOBALS
    if dbg:
        GLOBALS = {}

    action = Action(args)

    if args.test:
        trainset, testset = get_datasets(args)
        test(args, testset, action)
    elif args.apply:
        apply(args, action)
    else:
        trainset, testset = get_datasets(args)
        train(args, trainset, testset, action)

    return GLOBALS


def test(args, testset, action):
    if not torch.cuda.is_available():
        args.device = "cpu"
    args.device = torch.device(args.device)

    model = action.create_model()

    # action.try_transfer(model, args.pretrained)
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location="cpu"))

    model.to(args.device)
    model.eval()  # Batch norms etc. configured for testing mode.


    # Dataloader
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=args.workers
    )

    action.test_1(model, testloader, args.device, epoch=0)



def train(args, trainset, testset, action):
    if not torch.cuda.is_available():
        args.device = "cpu"
    args.device = torch.device(args.device)

    model = action.create_model()
    
    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad) 
    print('model parameter:', num_params)

    for m in model.modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, torch.nn.BatchNorm1d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)


    # action.try_transfer(model, args.pretrained)
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location="cpu"))
    model.to(args.device)

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])

    # dataloader
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    print('traindata', len(trainset))
    print('testdata', len(testset))

    # Optimizer
    min_loss = float("inf")
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(learnable_params, lr=1e-3)#, weight_decay= 1e-5
    elif args.optimizer == "RMSProp":
        optimizer = torch.optim.RMSprop(learnable_params, lr=0.001)#, weight_decay= 1e-5
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.001, momentum=0.9)#, weight_decay= 1e-5

    if checkpoint is not None:
        min_loss = checkpoint["min_loss"]
        optimizer.load_state_dict(checkpoint["optimizer"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.000001)
   
    # training
    LOGGER.debug("train, begin")
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_rotation_error, info = action.train_1(
            model, trainloader, optimizer, args.device, epoch
        )
        val_loss, val_rotation_error, info_1 = action.eval_1(
            model, testloader, args.device, epoch
        )
        LOGGER.info(
           info
        )
        LOGGER.info(
           info_1
        )
        scheduler.step()
   

        is_best = (1- info['r_ab_r2_score'] ) < min_loss
        min_loss = min((1- info['r_ab_r2_score'] ), min_loss)

        LOGGER.info(
            "epoch, %04d, train_loss=%f, train_rotation_error=%f, val_loss=%f, val_rotation_error=%f",
            epoch + 1,
            train_loss,
            train_rotation_error,
            val_loss,
            val_rotation_error,
        )

        snap = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "min_loss": min_loss,
            "optimizer": optimizer.state_dict(),
        }
        if is_best:
            save_checkpoint(snap, args.outfile, "snap_best")
            save_checkpoint(model.state_dict(), args.outfile, "model_best")

        save_checkpoint(snap, args.outfile, "snap_last")
        save_checkpoint(model.state_dict(), args.outfile, "model_last")

    LOGGER.debug("train, end")


def save_checkpoint(state, filename, suffix):
    torch.save(state, "{}_{}.pth".format(filename, suffix))


class Action:
    def __init__(self, args):
        self.experiment_name = args.pretrained

        self.transfer_from = args.transfer_from

        self.p0_zero_mean = True
        self.p1_zero_mean = True

        self.LOSS_TYPE = args.loss_type

        # SampleNet:
        self.ALPHA = args.alpha  # Sampling loss
        self.LMBDA = args.lmbda  # Projection loss
        self.GAMMA = args.gamma  # Inside sampling loss - linear.
        self.DELTA = args.delta  # Inside sampling loss - point cloud size factor.
        self.NUM_IN_POINTS = args.num_in_points
        self.NUM_OUT_POINTS = args.num_out_points
        self.BOTTLNECK_SIZE = args.bottleneck_size
        self.GROUP_SIZE = args.projection_group_size

        self.SKIP_PROJECTION = args.skip_projection
        self.SAMPLER = args.sampler

        self.TRAIN_SAMPLENET = args.train_samplenet
        self.TRAIN_PCRNET = args.train_pcrnet
        self.NUM_SAMPLED_CLOUDS = args.num_sampled_clouds

    def create_model(self):
        # Create Task network and load pretrained feature weights if requested
        pcrnet_model = pcrnet.PCRNet(input_shape="bnc")

        if self.TRAIN_PCRNET:
            pcrnet_model.requires_grad_(True)
            pcrnet_model.train()
        else:
            pcrnet_model.requires_grad_(False)
            pcrnet_model.eval()

        return pcrnet_model

    @staticmethod
    def try_transfer(model, path):
        if path is not None:
            model.load_state_dict(torch.load(path, map_location="cpu"))
            LOGGER.info(f"Model loaded from {path}")

    def valid_metric(self, rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred):
        rotations_ab = np.concatenate(rotations_ab, axis=0)
        translations_ab = np.concatenate(translations_ab, axis=0)
        rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
        translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
       
        eulers_ab = self.dcm2euler(rotations_ab)
        eulers_ab_pred = self.dcm2euler(rotations_ab_pred)
        r_ab_mse = np.mean((eulers_ab - eulers_ab_pred) ** 2)
        r_ab_rmse = np.sqrt(r_ab_mse)
        r_ab_mae = np.mean(np.abs(eulers_ab - eulers_ab_pred))
        t_ab_mse = np.mean((translations_ab - translations_ab_pred) ** 2)
        t_ab_rmse = np.sqrt(t_ab_mse)
        t_ab_mae = np.mean(np.abs(translations_ab - translations_ab_pred))
        r_ab_r2_score = r2_score(eulers_ab, eulers_ab_pred)
        t_ab_r2_score = r2_score(translations_ab, translations_ab_pred)

        info = {
                'r_ab_mse': r_ab_mse,
                'r_ab_rmse': r_ab_rmse,
                'r_ab_mae': r_ab_mae,
                't_ab_mse': t_ab_mse,
                't_ab_rmse': t_ab_rmse,
                't_ab_mae': t_ab_mae,
                'r_ab_r2_score': r_ab_r2_score,
                't_ab_r2_score': t_ab_r2_score}
        '''
        print(f'r_ab_mse= { r_ab_mse},'+
                f'r_ab_rmse= {r_ab_rmse},'+
                f'r_ab_mae={ r_ab_mae},'+
                f't_ab_mse= {t_ab_mse},'+
                f't_ab_rmse= {t_ab_rmse},'+
                f't_ab_mae= {t_ab_mae},'+
                f'r_ab_r2_score={ r_ab_r2_score},'+
                f't_ab_r2_score= {t_ab_r2_score}')
        '''
        return info

    def apply_dropout(self, m):
        if type(m) == torch.nn.Dropout:
           m.train()
    
    def freeze_bn(self, m):
        if isinstance(m, torch.nn.BatchNorm1d):
            m.eval()

    def train_1(self, model, trainloader, optimizer, device, epoch):
        vloss = 0.0
        gloss = 0.0

        count = 0
        model.train()

        rotations_ab = []
        translations_ab = []
        rotations_ab_pred = []
        translations_ab_pred = []

        for i, data in enumerate(tqdm(trainloader)):
            # Sample using one of the samplers:
            pcrnet_loss, pcrnet_loss_info = self.compute_pcrnet_loss(
                model, data, device, epoch
            )

            rotation_error = pcrnet_loss_info["rot_err"]
           
            loss = pcrnet_loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            vloss1 = loss.item()
            vloss += vloss1
            gloss1 = rotation_error.item()
            gloss += gloss1
            count += 1

            rotations_ab.append(pcrnet_loss_info['gt_transform'].rotate().detach().cpu().numpy())
            translations_ab.append(pcrnet_loss_info['gt_transform'].trans().detach().cpu().numpy())
            rotations_ab_pred.append(pcrnet_loss_info['est_transform'].rotate().detach().cpu().numpy())
            translations_ab_pred.append(pcrnet_loss_info['est_transform'].trans().detach().cpu().numpy())

        info =self.valid_metric(rotations_ab,translations_ab, rotations_ab_pred,translations_ab_pred )

        ave_vloss = float(vloss) / count
        ave_gloss = float(gloss) / count
        return ave_vloss, ave_gloss, info

    def eval_1(self, model, testloader, device, epoch):
        vloss = 0.0
        gloss = 0.0

        # Shift to eval mode for BN / Projection layers
        task_state = model.training
        model.eval()

        #model.apply(self.apply_dropout)
        model.apply(self.freeze_bn)
      

        count = 0

        rotations_ab = []
        translations_ab = []
        rotations_ab_pred = []
        translations_ab_pred = []

        with torch.no_grad():
            for i, data in enumerate(testloader):
                # Sample using one of the samplers:
                pcrnet_loss, pcrnet_loss_info = self.compute_pcrnet_loss(
                    model, data, device, epoch
                )

                rotation_error = pcrnet_loss_info["rot_err"]

                loss = pcrnet_loss 

                vloss1 = loss.item()
                vloss += vloss1
                gloss1 = rotation_error.item()
                gloss += gloss1
                count += 1

                rotations_ab.append(pcrnet_loss_info['gt_transform'].rotate().detach().cpu().numpy())
                translations_ab.append(pcrnet_loss_info['gt_transform'].trans().detach().cpu().numpy())
                rotations_ab_pred.append(pcrnet_loss_info['est_transform'].rotate().detach().cpu().numpy())
                translations_ab_pred.append(pcrnet_loss_info['est_transform'].trans().detach().cpu().numpy())

        ave_vloss = float(vloss) / count
        ave_gloss = float(gloss) / count

        # Shift back to training (?) mode for task and samppler
        model.train(task_state)
        
        info =self.valid_metric(rotations_ab,translations_ab, rotations_ab_pred,translations_ab_pred )

        return ave_vloss, ave_gloss, info

     
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

    def identity(self, batch_size):
        return torch.eye(3, 4)[None, ...].repeat(batch_size, 1, 1)


    def inverse(self, g):
        """ Returns the inverse of the SE3 transform

        Args:
            g: (B, 3/4, 4) transform

        Returns:
            (B, 3, 4) matrix containing the inverse

        """
        # Compute inverse
        rot = g[..., 0:3, 0:3]
        trans = g[..., 0:3, 3]
        inverse_transform = torch.cat([rot.transpose(-1, -2), rot.transpose(-1, -2) @ -trans[..., None]], dim=-1)

        return inverse_transform


    def concatenate(self,a, b):
        """Concatenate two SE3 transforms,
        i.e. return a@b (but note that our SE3 is represented as a 3x4 matrix)
        
        Args:
            a: (B, 3/4, 4) 
            b: (B, 3/4, 4) 

        Returns:
            (B, 3/4, 4)
        """

        rot1 = a[..., :3, :3]
        trans1 = a[..., :3, 3]
        rot2 = b[..., :3, :3]
        trans2 = b[..., :3, 3]
        
        rot_cat = rot1 @ rot2
        trans_cat = rot1 @ trans2[..., None] + trans1[..., None]
        concatenated = torch.cat([rot_cat, trans_cat], dim=-1)

        return concatenated


    def transform(self,g, a, normals=None):
        """ Applies the SE3 transform

        Args:
            g: SE3 transformation matrix of size ([1,] 3/4, 4) or (B, 3/4, 4)
            a: Points to be transformed (N, 3) or (B, N, 3)
            normals: (Optional). If provided, normals will be transformed

        Returns:
            transformed points of size (N, 3) or (B, N, 3)

        """
        R = g[..., :3, :3]  # (B, 3, 3)
        p = g[..., :3, 3]  # (B, 3)

        if len(g.size()) == len(a.size()):
            b = torch.matmul(a, R.transpose(-1, -2)) + p[..., None, :]
        else:
            raise NotImplementedError
            b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p  # No batch. Not checked

        if normals is not None:
            rotated_normals = normals @ R.transpose(-1, -2)
            return b, rotated_normals

        else:
            return b

    def compute_metrics(self, p1, p0,  gt_transforms_rotate, gt_transforms_trans
        , pred_transforms_rotate, pred_transforms_trans):
        """Compute metrics required in the paper
        """
        def square_distance(src, dst):
            return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

        with torch.no_grad():
          
            # Euler angles, Individual translation errors (Deep Closest Point convention)
            # TODO Change rotation to torch operations

            r_gt_euler_deg = self.dcm2euler(np.array(gt_transforms_rotate), seq='xyz')
            r_pred_euler_deg = self.dcm2euler(np.array(pred_transforms_rotate), seq='xyz')
            
            t_gt = np.array(gt_transforms_trans)
            t_pred =np.array( pred_transforms_trans)
            #print(r_gt_euler_deg)
            #print(r_pred_euler_deg)
            r_mse = np.mean((r_gt_euler_deg - r_pred_euler_deg) ** 2)
            r_rmse = np.sqrt(r_mse)
            r_mae = np.mean(np.abs(r_gt_euler_deg - r_pred_euler_deg))
            t_mse = np.mean((t_gt - t_pred) ** 2)
            t_rmse = np.sqrt(t_mse)
            t_mae = np.mean(np.abs(t_gt - t_pred))
            r_ab_r2_score = r2_score(r_gt_euler_deg, r_pred_euler_deg)
            t_ab_r2_score = r2_score(t_gt, t_pred)

            # Rotation, translation errors (isotropic, i.e. doesn't depend on error
            # direction, which is more representative of the actual error)
            #concatenated = self.concatenate(self.inverse(gt_transforms), pred_transforms)
            #rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
            #residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
            #residual_transmag = concatenated[:, :, 3].norm(dim=-1)

            # Modified Chamfer distance
            #src_transformed = se3.transform(pred_transforms, points_src)
            #ref_clean = points_raw
            #src_clean = se3.transform(se3.concatenate(pred_transforms, se3.inverse(gt_transforms)), points_raw)
            #dist_src = torch.min(square_distance(src_transformed, ref_clean), dim=-1)[0]
            #dist_ref = torch.min(square_distance(points_ref, src_clean), dim=-1)[0]
            #chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

            metrics = {
                'r_mse': r_mse,
                'r_rmse': r_rmse,
                'r_mae': r_mae,
                'r_ab_r2_score': r_ab_r2_score,
                't_mse': t_mse,
                't_rmse': t_rmse,
                't_mae': t_mae,
                't_ab_r2_score':t_ab_r2_score
                #'err_r_deg': to_numpy(residual_rotdeg),
                #'err_t': to_numpy(residual_transmag),
               # 'chamfer_dist': to_numpy(chamfer_dist)
            }

        return metrics


    def test_1(self, model, testloader, device, epoch):
        gt_transform_rotate = []
        gt_transform_trans = []
        gt_transform_scale = []
        est_transform_rotate = []
        est_transform_trans = []
        est_transform_scale = []

        model.apply(self.apply_dropout)
        model.apply(self.freeze_bn)

        with torch.no_grad():
            for i, data_and_shape in enumerate(tqdm(testloader)):
              
                data = data_and_shape[0:4]
                shape = data_and_shape[4]

                p0, p1, objects, igt = data
            
                p0 = p0.to(device)  # template
                p1 = p1.to(device)  # source
                objects = objects.to(device)  # source
                gt_transform = QuaternionTransform.from_dict(igt, device) #真实变换的p0
            
                #--------------------------------------训练生成器 --------------------------------------------#
                preT, _ = model (p0, p1)    #得到预测的变换
                est_transform = QuaternionTransform(preT) 
                pre_p0 = est_transform.apply_transform(p0)  #将p0根据预测变换
                gt_p0 = gt_transform.apply_transform(p0)

                np.savetxt(str(i)+'_gtp0.txt', np.column_stack((gt_p0.cpu().numpy()[0,:, 0],gt_p0.cpu().numpy()[0,:, 1],gt_p0.cpu().numpy()[0,:, 2])),fmt='%f %f %f',newline='\n' ) #保存为整数
                np.savetxt(str(i)+'_p0.txt', np.column_stack((p0.cpu().numpy()[0,:, 0],p0.cpu().numpy()[0,:, 1],p0.cpu().numpy()[0,:, 2])),fmt='%f %f %f',newline='\n' ) #保存为整数
                np.savetxt(str(i)+'_p1.txt', np.column_stack((p1.cpu().numpy()[0,:, 0],p1.cpu().numpy()[0,:, 1],p1.cpu().numpy()[0,:, 2])),fmt='%f %f %f',newline='\n' ) #保存为整数
                np.savetxt(str(i)+'_prep0.txt',np.column_stack((pre_p0.cpu().numpy()[0,:, 0],pre_p0.cpu().numpy()[0,:,1],pre_p0.cpu().numpy()[0,:, 2])),fmt='%f %f %f',newline='\n') #保存为整数

                gt_transform_rotate.append(gt_transform.rotate().squeeze(0).cpu().numpy())
                gt_transform_trans.append(gt_transform.trans().squeeze(0).cpu().numpy())
              
                est_transform_rotate.append(est_transform.rotate().squeeze(0).cpu().numpy())
                est_transform_trans.append(est_transform.trans().squeeze(0).cpu().numpy())
            
        metric = self.compute_metrics(p1, p0, gt_transform_rotate,gt_transform_trans, 
            est_transform_rotate, est_transform_trans)
        
        print(f"Experiment name: {self.experiment_name}")
        print(f"r_mse = {metric['r_mse']}")
        print(f"r_rmse = {metric['r_rmse']}")
        print(f"r_mae = {metric['r_mae']}")
        print(f"t_mse = {metric['t_mse']}") 
        print(f"t_rmse = {metric['t_rmse']}")  
        print(f"t_mae = {metric['t_mae']}")
        print(f"r_ab_r2_score = {metric['r_ab_r2_score']}")
        print(f"t_ab_r2_score = {metric['t_ab_r2_score']}")
    
    
    def compute_pcrnet_loss(self, model, data, device, epoch):
        p_source, p_target, objects,  igt = data
        p_source = p_source.to(device)  # source
        p_target = p_target.to(device)  # template
        igt_vec = igt['vec'].to(device) # igt: p0 -> p1

        predT0, _ = model(p_source, p_target)
       
        est_transform = QuaternionTransform(predT0)
        gt_transform = QuaternionTransform.from_dict(igt, device)

        rot_err, trans_err = est_transform.compute_errors(gt_transform)
      
        loss_p0_p1 = (torch.mean( rot_err)+ torch.mean(  trans_err))# 
       
        pcrnet_loss = loss_p0_p1 
        
        pcrnet_loss_info = {
            "rot_err": torch.mean(rot_err),
            "est_transform": est_transform,
            "gt_transform": gt_transform
        }

        return pcrnet_loss, pcrnet_loss_info

#get source and target points
def get_datasets(args):
    #numpy to tensor
    transforms = torchvision.transforms.Compose(
            [
                PointcloudToTensor(),
            ])
    
    if not args.test:
        traindata = ModelNetCls(
            args.num_in_points,
            transforms=transforms,
            train=True,
            download=False,
            folder=args.datafolder,
        )
        testdata = ModelNetCls(
            args.num_in_points,
            transforms=transforms,
            train=False,
            download=False,
            folder=args.datafolder,
        )

        train_repeats = max(int(50000 / len(traindata)), 1)
        print(train_repeats)
        #transformation
        trainset = QuaternionFixedDataset(args, traindata, repeat=train_repeats, seed=0,)
        testset = QuaternionFixedDataset(args, testdata, repeat=1, seed=0)
    else:
        testdata = ModelNetCls(
            args.num_in_points,
            transforms=transforms,
            train=False,
            download=False,
            cinfo=None,
            folder=args.datafolder,
            include_shapes=True,
        )
        trainset = None
        testset = QuaternionFixedDataset(args,testdata, repeat=1, seed=1)

    return trainset, testset


if __name__ == "__main__":
    ARGS = options(parser=sputils.get_parser())

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s:%(name)s, %(asctime)s, %(message)s",
        filename=f"{ARGS.outfile}.log",
    )
    LOGGER.debug("Training (PID=%d), %s", os.getpid(), ARGS)

    _ = main(ARGS)
    LOGGER.debug("done (PID=%d)", os.getpid())
