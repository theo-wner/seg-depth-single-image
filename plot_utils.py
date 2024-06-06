import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.cm

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
def visualize_img_gts(image, gt_label, gt_depth, filename='test.png'):
    # --------------------------------------------------------------------------------------------
    # place subplots
    # --------------------------------------------------------------------------------------------
    plt.figure(figsize=(16, 4.5), frameon=False)

    # Leave everything as it is!!!
    # If then only adjust the wspace value!!!
    plt.subplots_adjust(left=0,
                        bottom=0,
                        right=0.79,
                        top=1,
                        wspace=0.05,
                        hspace=0.0)
    
    # --------------------------------------------------------------------------------------------
    # Image
    # --------------------------------------------------------------------------------------------
    # Convert Image from Tensor to Image
    image = image.permute(1, 2, 0).numpy()
    plt.subplot(1, 3, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Label Preprocessing
    # --------------------------------------------------------------------------------------------
    gt_label = map_40_to_13(gt_label)

    # Definde Labels and Colors as Dictionary
    labels_and_colors = get_labels_and_colors()

    # Create Colormap from Dictionary
    cmap = mcolors.ListedColormap(list(labels_and_colors.values()))

    # --------------------------------------------------------------------------------------------
    # Gt Label
    # --------------------------------------------------------------------------------------------
    # Convert Mask from Tensor to Image
    gt_label = gt_label.squeeze().numpy()
    # Set Unlabeled Pixels to Value 14 (For the colormap)
    gt_label[gt_label == 255] = 14

    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.yticks([])
    #plt.imshow(image)
    plt.imshow(gt_label, cmap=cmap, vmin=0, vmax=14)#, alpha=0.5)
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Gt Depth
    # --------------------------------------------------------------------------------------------
    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    cmap = matplotlib.cm.get_cmap('plasma_r')
    norm = plt.Normalize(vmin=0, vmax=10)
    plt.imshow(gt_depth.squeeze().numpy(), cmap=cmap, norm=norm)
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Legend
    # --------------------------------------------------------------------------------------------
    legend_elements = [mpatches.Patch(facecolor=labels_and_colors[label],
                             edgecolor='black',
                             label=label) for label in labels_and_colors]
    plt.legend(handles=legend_elements,
               loc='center left',
               bbox_to_anchor=(1.35, 0.5))
    
    # Set Legend Title
    #plt.gca().get_legend().set_title('Annotationen')

    # Set the legend font size
    #plt.gca().get_legend().get_title().set_fontsize('medium')
    
    # Make legend bold
    #plt.setp(plt.gca().get_legend().get_title(), fontweight='bold')

    # Set the font size of the labels and font type
    for label in plt.gca().get_legend().get_texts():
        label.set_fontsize('xx-large')
        #label.set_fontfamily('serif')

    # Make the font of my legend look like latex
    #plt.gca().get_legend().get_title().set_fontfamily('serif')

    # --------------------------------------------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------------------------------------------
    # Colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), 
                    ax=plt.gcf().get_axes(), 
                    orientation='vertical', 
                    pad=0.02,
                    fraction=0.0165)
    cbar.set_label('Tiefe (m)', size='xx-large')
    cbar.ax.tick_params(labelsize='xx-large')

    # --------------------------------------------------------------------------------------------
    # Show Plot
    # --------------------------------------------------------------------------------------------
    plt.show()