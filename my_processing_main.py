# import os
# import cv2
# import numpy as np
# from scipy import ndimage
# import torch
# import trimesh
# from pytorch3d.utils import opencv_from_cameras_projection
# from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
# from pytorch3d.renderer import PerspectiveCameras
# from scipy.spatial.transform import Rotation as R
#
#
# def post_process(path):
#     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#     mask = cv2.imread(path.replace('composition', 'seg_mask'), cv2.IMREAD_UNCHANGED)[:, :, 2:3]
#     mask = (mask == 90)
#     mask = (mask > 0).astype(np.int32)
#     mask = ndimage.median_filter(mask, size=5)
#     mask = (ndimage.binary_dilation(mask, iterations=3) > 0).astype(np.uint8)
#     alpha = img[:, :, 3:4] / 1.0
#     img *= (1 - mask)
#     cv2.imwrite(path, img.astype(np.uint8))
#     alpha = ndimage.median_filter(alpha, size=5)
#     alpha = np.where(mask, np.zeros_like(alpha), alpha)
#     alpha[alpha > 127.5] = 255
#     alpha = alpha.astype(np.uint8)
#     img[:, :, 3:4] = alpha
#     cv2.imwrite(path.replace('composition', 'images'), img)
#
#
# def save_canonical(out_dir):
#     canon = os.path.join(out_dir, "canonical.obj")
#     rotvec = np.zeros(3)
#     rotvec[0] = 12.0 * np.pi / 180.0
#
#     focal_length = torch.tensor([14.0]).to('cuda')
#     # TODO: insert the real image_size
#     image_size = 512
#
#     # TODO: validate principal_point - use face-animation's create_camera(camera_focal_point: Tensor, img_size: int, device: str)
#     cameras = PerspectiveCameras(
#         device='cuda',
#         principal_point=principal_point,
#         focal_length=focal_length,
#         R=rotation_6d_to_matrix(torch.eye(3)[None]),
#         image_size=(image_size,)
#     )
#     shape = np.load()  # load my shape here (from fitting)
#
#     jaw = matrix_to_rotation_6d(torch.from_numpy(R.from_rotvec(rotvec).as_matrix())[None, ...].cuda()).float()
#
#     # TODO: add face-animation's Flame
#     flame_cfg = FlameConfig(shape_params=100, expression_params=50)
#     flame = Flame(config=flame_cfg, device=device).eval()
#     faces = flame.get_faces()[None]
#     vertices = flame(cameras=torch.inverse(cameras.R), shape_params=shape, jaw_pose_params=jaw)[0].detach()
#     trimesh.Trimesh(faces=faces, vertices=vertices[0].cpu().numpy(), process=False).export(canon)
#
#
# # TODO: pass flame, image_size (int!!) and cameras like in the save_canonical() above
# def save_checkpoint(frame_id, out_dir, image_size, cameras):
#     opencv = opencv_from_cameras_projection(cameras, image_size)
#
#     focal_length = torch.tensor([14.0]).to('cuda').detach().cpu().numpy()
#     translation = torch.zeros(1, 3).numpy()
#     # TODO: use same as in face-animation
#     principal_point =
#     # TODO: load flame from gt, compare to what we get when loading the params in the pipe of GaussianBlendSahpes
#     # TODO: global_step - what should this be? compare to the loading in the pipe of GaussianBlendSahpes
#     frame = {
#         'flame': {
#             'exp': exp.clone().detach().cpu().numpy(),
#             'shape': shape.clone().detach().cpu().numpy(),
#             'tex': tex.clone().detach().cpu().numpy(),
#             'sh': sh.clone().detach().cpu().numpy(),
#             'eyes': eyes.clone().detach().cpu().numpy(),
#             'eyelids': eyelids.clone().detach().cpu().numpy(),
#             'jaw': jaw.clone().detach().cpu().numpy()
#         },
#         'camera': {
#             'R': cameras.R.detach().cpu().numpy(),
#             't': translation,
#             'fl': focal_length,
#             'pp': principal_point,
#         },
#         'opencv': {
#             'R': opencv[0].clone().detach().cpu().numpy(),
#             't': opencv[1].clone().detach().cpu().numpy(),
#             'K': opencv[2].clone().detach().cpu().numpy(),
#         },
#         'img_size': image_size,
#         'frame_id': frame_id,
#         'global_step': global_step
#     }
#
#     # vertices, _, _ = flame(
#     #     cameras=torch.inverse(cameras.R),
#     #     shape_params=shape,
#     #     expression_params=exp,
#     #     eye_pose_params=eyes,
#     #     jaw_pose_params=jaw,
#     #     eyelid_params=eyelids
#     # )
#
#     # v = vertices[0].cpu().numpy()
#
#     # trimesh.Trimesh(faces=flame.get_faces()[None], vertices=v, process=False).export(f'{mesh_folder}/{frame_id}.ply')
#     torch.save(frame, f'{out_dir}/checkpoint/{frame_id}.frame')
#
#
# # video_dir = ''
# # mask_dir = '/data/repos/RobustVideoMatting/outputs/viktoriia-speaking/alpha'
# # out_dir = ''
# # out_orig_frames_dir = ''
# #
# # for i, file_name in enumerate(os.listdir(mask_dir)):
# #     mask_path = os.path.join(mask_dir, file_name)
# #     frame_path = os.path.join(video_dir, file_name)
# #     mask = (cv2.imread(mask_path, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 > 0).float()
# #     frame = cv2.imread(mask_path, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
# #     predicted_images = frame * mask + np.ones_like(mask) - mask
# #
# #     frame_id = str(i).zfill(5)
# #     os.makedirs(out_dir, exist_ok=True)
# #     os.makedirs(out_orig_frames_dir, exist_ok=True)
# #     cv2.imwrite('{}/{}.jpg'.format(out_dir, frame_id), predicted_images)
# #     cv2.imwrite('{}/{}.png'.format(out_orig_frames_dir, frame_id), frame)
#
#
# # step 1 - run the RobustVideoMatting inference script to get bg removal - create "matted" dir
# # python inference.py --variant resnet50 --checkpoint model/rvm_resnet50.pth --device cuda:0 --input-source /data/repos/RobustVideoMatting/inputs/viktoriia-speaking/viktoriia-speaking-trimmed.mp4 --output-type png_sequence --output-composition /data/repos/RobustVideoMatting/outputs/viktoriia-speaking/composition --output-foreground /data/repos/RobustVideoMatting/outputs/viktoriia-speaking/foreground --output-alpha /data/repos/RobustVideoMatting/outputs/viktoriia-speaking/alpha
#
# # step 2 - run face-parsing (using BiSeNet) - at this point they assume we get "90" as the clothes value - create "seg_mask" dir
# # input dir is: /data/repos/RobustVideoMatting/outputs/viktoriia-speaking/composition
# # output dir is: /data/repos/RobustVideoMatting/outputs/viktoriia-speaking/seg_mask
#
# # step 3 - run the post_process (defined in this script) - to get the final "floating" head - create "images" dir
# # input file path is: /data/repos/RobustVideoMatting/outputs/viktoriia-speaking/composition/1499.png
# # output file path is: /data/repos/RobustVideoMatting/outputs/viktoriia-speaking/images/1499.png
#
# # step 4 - run save_canonical() to get "canonical.obj" (generated by the tracker)
#
# # step 5 - run save_checkpoint() to get the pre-frame checkpoint (generated by the tracker)
#
# # step 6 - call GaussianBlendShapes on my own dataset
