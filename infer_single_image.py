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

if __name__ == '__main__':

    # Set the verbosity of the transformers library to error
    transformers.logging.set_verbosity_error()

    # Load model
    model = SegDepthFormer().to('cpu')
    checkpoint = torch.load(f'./models/{config.BACKBONE}/checkpoints/epoch=399-step=40000.ckpt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Disable gradients
    model.eval()

    # Load image from webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = image.resize((640, 480))
    image = TF.to_tensor(image).unsqueeze(0).to('cpu')


    # Load image
    #image = np.array(Image.open('./data/image/test/00001.png'))
    #image = TF.to_tensor(image).unsqueeze(0).to('cpu')

    # Infer
    with torch.no_grad():
        seg_logits, depth_preds = model(image)
        upsampled_seg_logits = torch.nn.functional.interpolate(seg_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        seg_preds = torch.softmax(upsampled_seg_logits, dim=1)
        seg_preds = torch.argmax(seg_preds, dim=1)
        
        depth_preds = F.relu(depth_preds, inplace=True)
        depth_preds = torch.nn.functional.interpolate(depth_preds, size=image.shape[-2:], mode="bilinear", align_corners=False)
        depth_preds = depth_preds.squeeze(1)

        image = image.squeeze(0)

        # Visualize
        visualize_img_gts(image, seg_preds, depth_preds, filename='test.png')

