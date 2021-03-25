import torch

from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from display_utils import display_model


if __name__ == '__main__':
    cuda = False
    batch_size = 1

    # create the SMPL layer
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='neutral',
        model_root='smplpytorch/native/models')

    # generate random pose and shape parameters

    # poses
    # pose_params = torch.rand(batch_size, 72) * 0.2 # normal
    pose_params = torch.rand(batch_size, 72) * 0.001 # t-pose

    # shapes
    shape_params = torch.rand(batch_size, 10) * 0.03 # normal
    # shape_params = torch.rand(batch_size, 10) * 0.001 # fatter
    # shape_params = torch.rand(batch_size, 10) * 10 # skinnier

    print('pose_params:', pose_params)
    print('shape_params:', shape_params)

    # GPU mode
    if cuda:
        pose_params = pose_params.cuda()
        shape_params = shape_params.cuda()
        smpl_layer.cuda()

    # forward from the SMPL layer
    verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)

    # draw output vertices and joints
    display_model(
        {'verts': verts.cpu().detach(),
         'joints': Jtr.cpu().detach()},
        model_faces=smpl_layer.th_faces,
        with_joints=True,
        kintree_table=smpl_layer.kintree_table,
        savepath='image.png',
        show=True)
