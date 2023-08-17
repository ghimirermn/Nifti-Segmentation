import argparse
import monai.networks.nets
import torch
import numpy as np
import monai
import nibabel as nib
from skimage.transform import resize
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    model = monai.networks.nets.VNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        act=('elu', {'inplace': True}),
        dropout_prob=0.5,
        dropout_dim=3,
        bias=False
    )
    model.load_state_dict(torch.load(args.model_weight))
    model.to(device)
    model.eval()

    # Load and preprocess the input image
    image = nib.load(args.image_path).get_fdata()

    if args.normalization is True:
        image = (image - image.min()) / (image.max() - image.min())

    if args.image_resize is not None:
        shape = args.image_resize
        image = resize(image, (shape, shape, shape), mode='constant')

    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image)

    output = output.cpu().squeeze().numpy()

    #creating output name
    input_filename = os.path.basename(args.image_path)
    output_filename = os.path.splitext(input_filename)[0] + ".gz"
    output_path = os.path.join(args.output_path, output_filename)

    #saving result
    output_image = nib.Nifti1Image(output, np.eye(4))
    nib.save(output_image, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment an image using a pre-trained VNet model.")
    parser.add_argument("--image_path", type=str, default="", help="Path to the input NIfTI image.")
    parser.add_argument("--model_weight", type=str, default= "", help="Path to the pre-trained model weight.")
    parser.add_argument("--output_path", type=str, default=".", help="Path to save the segmentation result.")
    parser.add_argument("--image_resize", type=int, default=None, help="Use the image size used during training.")
    parser.add_argument("--normalization", type=str, default=True,help="True if the training data has been normalized")
    args = parser.parse_args()
    main(args)
