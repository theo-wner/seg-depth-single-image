'''Script for infering a single image using a trained model'''

from model import SegDepthFormer
import config
from plot_utils import *
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import transformers
from PIL import Image
import cv2
import open3d as o3d

if __name__ == '__main__':

    # Set the verbosity of the transformers library to error
    transformers.logging.set_verbosity_error()

    # Load model
    model = SegDepthFormer().to('cpu')
    checkpoint = torch.load(f'./models/{config.BACKBONE}/checkpoints/epoch=399-step=40000.ckpt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Disable gradients
    model.eval()

    while True:
        key = input("Press 'c' to capture image, 'q' to quit: ")
        if key == 'q':
            break
        elif key == 'c':
            pass

        # Load image from webcam
        cap = cv2.VideoCapture(4) # 0 is Laptop webcam, 4 is external webcam
        ret, frame = cap.read()
        cap.release()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((640, 480))
        image = TF.to_tensor(image).unsqueeze(0).to('cpu')

        # Infer
        with torch.no_grad():
            seg_logits, depth_preds = model(image)
            upsampled_seg_logits = torch.nn.functional.interpolate(seg_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
            seg_preds = torch.softmax(upsampled_seg_logits, dim=1)
            seg_preds = torch.argmax(seg_preds, dim=1)
            
            depth_preds = F.relu(depth_preds, inplace=True)
            depth_preds = torch.nn.functional.interpolate(depth_preds, size=image.shape[-2:], mode="bilinear", align_corners=False)
            depth_preds = depth_preds.squeeze(1).numpy()

            # Visualize
            image_out, seg_out, depth_out = visualize_img_gts(image, seg_preds, depth_preds)











































# Experimental: Visualize point cloud
'''
# Convert to Open3D Image
seg_out = np.array(seg_out).astype(np.uint8)
depth_preds = np.moveaxis(depth_preds, 0, -1)

seg_out = o3d.geometry.Image(seg_out)
depth_preds = o3d.geometry.Image(depth_preds)

# Create dummy camera intrinsics of a 640x480 webcam image
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 400,400, 320, 240)

# Calculate point cloud
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(seg_out, depth_preds, convert_rgb_to_intensity=False)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

# flip the orientation, so it looks upright, not upside-down
pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

vis = o3d.visualization.Visualizer()
vis.create_window()

vis.add_geometry(pcd)

opt = vis.get_render_option()
opt.point_size = 1

vis.run()
vis.destroy_window()
'''


