import matplotlib.pyplot as plt

def extract_patches(img, patch_size):
    '''
        extract all patch_size x patch_size patches from img
    '''
    h, w = img.shape # get width and height
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            # get rows i -> i + patch_size (not inclusive)
            # get colums j -> j + patch_size (not inclusive)
            patch = img[i:i+patch_size, j:j+patch_size]
            # make sure that the patch is square
            if patch.shape == (patch_size, patch_size):
                patches.append(patch)
    return patches

def show_patches_grid(patches, cols=16):
    """
    Display patches in a grid.

    Args:
        patches (list): List of image patches.
        cols (int, optional): Number of columns in the grid. Defaults to 16.
    """
    n_patches = len(patches)
    rows = (n_patches + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten()

    # global min and max across all patches for color bar
    gray_patches = [p for p in patches if p.ndim == 2]
    vmin = min(p.min() for p in gray_patches) if gray_patches else 0
    vmax = max(p.max() for p in gray_patches) if gray_patches else 1

    for ax, patch in zip(axes, patches):
        if patch.ndim == 2:
            ax.imshow(patch, cmap='gray', vmin=vmin, vmax=vmax)
        else:
            ax.imshow(patch)
        ax.axis('off')

    for ax in axes[len(patches):]:
        ax.axis('off')

    plt.tight_layout()
    fig.savefig("grid_patches.svg")
    plt.show()