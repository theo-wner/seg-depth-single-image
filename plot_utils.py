import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.cm
from PIL import Image

'''
Maps 40 classes to 13 classes for Segmentation
'''
def map_40_to_13(mask):
    mapping = {0:11, 1:4, 2:5, 3:0, 4:3, 5:8, 6:9, 7:11, 8:12,	9:5, 10:7, 11:5, 12:12, 13:9, 14:5, 15:12,	
               16:5, 17:6, 18:6, 19:4, 20:6, 21:2, 22:1, 23:5, 24:10, 25:6, 26:6, 27:6, 28:6, 29:6,	30:6, 
               31:5, 32:6, 33:6, 34:6, 35:6, 36:6, 37:6, 38:5, 39:6, 255:255}
    
    mask = mask.squeeze().numpy().astype(int)

    for r, c in np.ndindex(mask.shape):
        mask[r, c] = mapping[mask[r, c]]

    return torch.tensor(mask, dtype=torch.long).unsqueeze(0)

'''
Returns a dictionary of labels and colors for the segmentation
'''
def get_labels_and_colors():
    labels_and_colors = {'Bett' : (0,0,1),
                         'Bücher' : (0.9137,0.3490,0.1882),
                         'Decke' : (0, 0.8549, 0),
                         'Stuhl' : (0.5843,0,0.9412),
                         'Fußboden' : (0.8706,0.9451,0.0941),
                         'Möbel' : (1.0000,0.8078,0.8078),
                         'Objekte' : (0,0.8784,0.8980),
                         'Bild' : (0.4157,0.5333,0.8000),
                         'Sofa' : (0.4588,0.1137,0.1608),
                         'Tisch' : (0.9412,0.1373,0.9216),
                         'Fernseher' : (0,0.6549,0.6118),
                         'Wand' : (0.9765,0.5451,0),
                         'Fenster' : (0.8824,0.8980,0.7608),
                         'Nicht annotiert' : (1,1,1)}
    return labels_and_colors


'''
Creates a plot of the image, gt labels and gt depth maps
'''
def visualize_img_gts(image, seg_preds, depth_preds):

    # Place subplots
    plt.figure(figsize=(16, 4.5), frameon=False)
    plt.subplots_adjust(left=0,
                        bottom=0,
                        right=0.79,
                        top=1,
                        wspace=0.05,
                        hspace=0.0)
    
    # Image
    image = image.squeeze(0).permute(1, 2, 0).numpy()

    plt.subplot(1, 3, 1)
    plt.xticks([])
    plt.yticks([])
    image_out = plt.imshow(image)
    image_out = image_out.get_array()
    image_out = Image.fromarray((image_out * 255).astype('uint8'))
    plt.axis('off')

    # Label Preprocessing
    seg_preds = map_40_to_13(seg_preds)
    labels_and_colors = get_labels_and_colors()
    cmap_seg = mcolors.ListedColormap(list(labels_and_colors.values()))

    # Segmentation Prediction
    seg_preds = seg_preds.squeeze().numpy()
    seg_preds[seg_preds == 255] = 14

    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.yticks([])
    seg_out = plt.imshow(seg_preds, cmap=cmap_seg, vmin=0, vmax=14)
    seg_out = np.array(seg_out.get_array())
    seg_out = cmap_seg(seg_out)
    seg_out = seg_out[:, :, :3]
    seg_out = Image.fromarray((seg_out * 255).astype('uint8'))
    plt.axis('off')

    # Depth Prediction
    depth_preds = depth_preds.squeeze()

    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    cmap_depth = matplotlib.cm.get_cmap('plasma_r')
    norm = plt.Normalize(vmin=0, vmax=10)
    depth_out = plt.imshow(depth_preds, cmap=cmap_depth, norm=norm)
    depth_out = depth_out.get_array()
    # Normalize depth map
    depth_out = (depth_out - np.min(depth_out)) / (np.max(depth_out) - np.min(depth_out))
    depth_out = cmap_depth(depth_out)
    depth_out = Image.fromarray((depth_out * 255).astype('uint8'))
    plt.axis('off')

    # Legend
    legend_elements = [mpatches.Patch(facecolor=labels_and_colors[label],
                             edgecolor='black',
                             label=label) for label in labels_and_colors]
    plt.legend(handles=legend_elements,
               loc='center left',
               bbox_to_anchor=(1.35, 0.5))
    
    for label in plt.gca().get_legend().get_texts():
        label.set_fontsize('xx-large')

    # Colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_depth, norm=norm), 
                    ax=plt.gcf().get_axes(), 
                    orientation='vertical', 
                    pad=0.02,
                    fraction=0.0165)
    cbar.set_label('Tiefe (m)', size='xx-large')
    cbar.ax.tick_params(labelsize='xx-large')

    # Show Plot
    plt.show()

    return image_out, seg_out, depth_out