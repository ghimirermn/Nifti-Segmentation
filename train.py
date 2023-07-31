import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import nibabel as nib
import numpy as np
from dataset.dataset import NiftiDataset
import argparse
import cv2
from tqdm import tqdm
import monai

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(model, criterion):
    # for calculation of metrics
    model.eval()
    test_loss = 0.0
    dice_metric = monai.metrics.DiceMetric(include_background=False, reduction="mean")

    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)

            test_outputs = model(test_inputs)

            test_outputs = torch.round(torch.sigmoid(test_outputs))

            test_loss += criterion(test_outputs, test_labels)

            # Compute Dice score
            dice_metric(y_pred=test_outputs, y=test_labels)

        test_loss /= len(test_loader)
        dice_score = dice_metric.aggregate()

        mean_dice = dice_score.item()

        train_log.write("Epoch [{}/{}], Test Loss: {:.4f}, Mean Dice: {:.4f}\n".format(epoch, opt.epoch, test_loss,
                                                                                       mean_dice))
        train_log.write("\n")

        writer.add_scalar('Test Loss', test_loss, epoch)
        writer.add_scalar('Mean Dice', mean_dice, epoch)

        print("Epoch [{}/{}], Test Loss: {:.4f}, Mean Dice: {:.4f}".format(epoch, opt.epoch, test_loss, mean_dice))

    return test_loss, mean_dice


def train(train_loader, test_loader, model, optimizer, criterion, epoch):
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Epoch {}/{}".format(epoch, opt.epoch))

    for inputs, labels in progress_bar:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        progress_bar.set_postfix({'Per Batch Train Loss': loss.item()})

    train_loss /= len(train_loader)
    train_log.write("Epoch [{}/{}], Train Loss: {:.4f}\n".format(epoch, opt.epoch, train_loss))

    writer.add_scalar('Per Epoch Train Loss', train_loss, epoch)
    print("Epoch [{}/{}], Train Loss : {:.4f}".format(epoch, opt.epoch, train_loss))

    if epoch % opt.save_interval == 0:
        checkpoint_path = os.path.join(save_path, "epoch")
        torch.save(model.state_dict(), checkpoint_path + '-%d.pth' % epoch)

        test(model, criterion)

    if epoch % opt.test_save == 0:
        output_folder = os.path.join("test_outputs", opt.task_name, f"epoch {epoch}")
        os.makedirs(output_folder, exist_ok=True)

        model.eval()
        with torch.no_grad():
            for k, (test_inputs, _) in enumerate(tqdm(test_loader, desc="Saving Images")):
                test_inputs = test_inputs.to(device)
                test_outputs = model(test_inputs)
                output = test_outputs.cpu().numpy()
                output = np.squeeze(output)
                original_image_path = test_dataset.get_image_path(k)
                original_image = nib.load(original_image_path)
                original_shape = original_image.shape

                resized_output = np.zeros(original_shape)
                for j in range(output.shape[0]):
                    resized_output[j] = cv2.resize(output[j], original_shape[1:], interpolation=cv2.INTER_LINEAR)

                output_path = os.path.join(output_folder, os.path.basename(original_image_path))
                output_image = nib.Nifti1Image(resized_output.astype(np.float32), original_image.affine)
                nib.save(output_image, output_path)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', type=str, default="3d Segmentation", help="task name")
    parser.add_argument('--epoch', type=int, default=300, help="number of epochs")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=2, help="training batch size")
    parser.add_argument('--trainsize', type=int, default=None, help="training dataset resize. The script can resize "
                                                                    "but the the training process becomes long")
    parser.add_argument('--save_interval', type=int, default=1, help="interval for saving model checkpoints")
    parser.add_argument('--test_save', type=int, default=1, help="interval to save results from test file")
    parser.add_argument('--clip', type=float, default=0.5, help="gradient clipping margin")
    parser.add_argument('--normalize', type=str, default=True, help="normalizing the input data")
    parser.add_argument('--resume_training', type=str, default=None, help="resume training from last checkpoint")
    parser.add_argument('--train_path', type=str,
                        default=None)
    parser.add_argument('--test_path', type=str,
                        default=None)

    opt = parser.parse_args()

    # monai is very convenient, from loading models to losses to various metrics.
    model = monai.networks.nets.VNet(spatial_dims=3, in_channels=1, out_channels=1,
                                     act=('elu', {'inplace': True}), dropout_prob=0.5, dropout_dim=3, bias=False)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    criterion = monai.losses.DiceCELoss(sigmoid=True)

    if opt.resume_training is not None:
        # load checkpoint
        model.load_state_dict(torch.load(opt.resume_training))

    train_image_folder = os.path.join(opt.train_path, "image")
    train_label_folder = os.path.join(opt.train_path, "label")

    test_image_folder = os.path.join(opt.test_path, "image")
    test_label_folder = os.path.join(opt.test_path, "label")

    train_dataset = NiftiDataset(train_image_folder, train_label_folder, resize_shape=opt.trainsize,
                                 normalize=opt.normalize)
    test_dataset = NiftiDataset(test_image_folder, test_label_folder, resize_shape=opt.trainsize,
                                normalize=opt.normalize)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create a SummaryWriter for logging
    log_dir = os.path.join("logs", opt.task_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Create the log file to record parameters and results
    log_file = os.path.join("logs", opt.task_name, "log.txt")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    train_log = open(log_file, "w")
    train_log.write("Arguments:\n")
    for arg, value in vars(opt).items():
        train_log.write("{}: {}\n".format(arg, value))
    train_log.write("\n")
    train_log.close()

    # Create a directory to save checkpoints
    save_path = os.path.join("snapshots", opt.task_name)
    os.makedirs(save_path, exist_ok=True)

    # saving test images at user specified intervals
    for epoch in range(1, opt.epoch + 1):
        train_log = open(log_file, "a")
        train(train_loader, test_loader, model, optimizer, criterion, epoch)

    train_log.close()
