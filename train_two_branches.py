import os
import pprint
import numpy as np
import tqdm
from utils.arg_parsing import parse_args
from utils.gpu_selection_utils import set_gpu_visible_devices
from datetime import datetime
import sys

main_start = datetime.now()
dt_string = main_start.strftime("%d/%m/%Y %H:%M:%S")
print("Start main() date and time =", dt_string)

args = parse_args()
set_gpu_visible_devices(num_gpus_to_use=args.num_gpus_to_use)


from utils.storage import (
    build_experiment_folder,
    save_checkpoint,
    restore_model,
    restore_model_from_path, save_metrics_dict_in_pt,
)

import random
import glob
import tarfile
import nibabel as nib
import tifffile as tif
import torchvision.transforms.functional as TF
from scipy import ndimage
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from utils.metric_tracking import (
    MetricTracker,
    compute_accuracy,
    plot_single_metrics,
)
from utils.custom_transforms import SliceSamplingUniform
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.model_architectures import *

args = parse_args()

height = 128
width = 128
channels = 1
slices = 11



def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    # volume = resize_volume(volume)
    return volume



transform = transforms.Compose(
    [
        SliceSamplingUniform(total_slices=slices),
        torchvision.transforms.Resize((int(height), int(width))),
    ]
)


class_0 = []
root_0 = os.path.join(args.dataset_root_folder, 'CT-0')
for x in os.listdir(root_0):
    class_0.append(os.path.join(root_0, x))

class_1 = []
root_1 = os.path.join(args.dataset_root_folder, 'CT-234')
for x in os.listdir(root_1):
    class_1.append(os.path.join(root_1, x))


images = class_0 + class_1

y = np.concatenate([np.zeros(len(class_0), dtype=int), np.ones(len(class_1), dtype=int)])

X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.15, stratify=y, random_state=args.seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.17647,
                                                  stratify=y_train, random_state=args.seed)

saliency_root = os.path.join(args.dataset_root_folder, 'saliency_maps')

class CustomImageDataset(Dataset):
    def __init__(self, images,labels, process, transform, flipping = False,target_transform=None):
        self.img_labels = labels
        self.image_list = images
        self.transform = transform
        self.process = process
        self.target_transform = target_transform
        self.flipping = flipping


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        name = os.path.basename(img_path)
        sal_path = os.path.join(saliency_root, f'{name}_AVG_gifmap.tif')
        sal = tif.imread(sal_path)
        # print('sal.shape')
        sal = sal.squeeze(0)
        sal = torch.tensor(sal)
        sal = sal.permute(2, 0, 1).unsqueeze(1)
        sal= self.transform(sal)
        # print('sal', sal.shape)

        image = self.process(img_path)
        image = torch.tensor(image)
        image = image.permute(2, 0, 1).unsqueeze(1)
        image = self.transform(image)
        # print('image', image.shape)

        if self.flipping:
            if random.random() > 0.5:
                image = TF.hflip(image)
                sal  = TF.hflip(sal)
            if random.random() > 0.5:
                image = TF.vflip(image)
                sal  = TF.vflip(sal)

        label = self.img_labels[idx]

        if self.target_transform:
            label = self.target_transform(label)
        return image,sal, label,img_path

train_dataset = CustomImageDataset(images = X_train, labels = y_train, process = process_scan, transform = transform, flipping = False)
val_dataset = CustomImageDataset(images = X_val, labels = y_val, process = process_scan, transform = transform, flipping=False)
test_dataset = CustomImageDataset(images = X_test, labels = y_test, process = process_scan, transform = transform, flipping=False)


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=True)




torch.manual_seed(args.seed)
np.random.seed(args.seed)  # set seed
random.seed(args.seed)


device = (
    torch.cuda.current_device()
    if torch.cuda.is_available() and args.num_gpus_to_use > 0
    else "cpu"
)
print(
    "Device: {} num_gpus: {}  torch.cuda.is_available() {}".format(
        device, args.num_gpus_to_use, torch.cuda.is_available()
    )
)
args.device = device

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

# Save a snapshot of the current state of the code.

saved_models_filepath, logs_filepath, images_filepath = build_experiment_folder(
    experiment_name=args.experiment_name,
    log_path=args.logs_path + "/" + "epochs_" + str(args.max_epochs),
)

snapshot_code_folder = saved_models_filepath
if not os.path.exists(snapshot_code_folder):
    os.makedirs(snapshot_code_folder)
snapshot_filename = "{}/snapshot.tar.gz".format(snapshot_code_folder)
filetypes_to_include = [".py"]
all_files = []
for filetype in filetypes_to_include:
    all_files += glob.glob("**/*.py", recursive=True)
with tarfile.open(snapshot_filename, "w:gz") as tar:
    for file in all_files:
        tar.add(file)
print(
    "saved_models_filepath: {} logs_filepath: {} images_filepath: {}".format(
        saved_models_filepath, logs_filepath, images_filepath
    )
)

print("-----------------------------------")
pprint.pprint(args, indent=4)
print("-----------------------------------")

if args.model =='ACAT':
    model = resnet50_ACAT(block=Bottleneck,
                              layers=[3, 4, 6, 3])
    best_model = resnet50_ACAT(block=Bottleneck,
                                   layers=[3, 4, 6, 3])
elif args.model =='SMIC':
    model = resnet50_SMIC(block=Bottleneck,
                          layers=[3, 4, 6, 3])
    best_model = resnet50_SMIC(block=Bottleneck,
                               layers=[3, 4, 6, 3])
elif args.model == 'HSM':
    model = resnet50_HSM(block=Bottleneck,
                          layers=[3, 4, 6, 3])
    best_model = resnet50_HSM(block=Bottleneck,
                               layers=[3, 4, 6, 3])
elif args.model == 'SalClassNet':
    model = resnet50_SalClassNet(block=Bottleneck,
                          layers=[3, 4, 6, 3])
    best_model = resnet50_SalClassNet(block=Bottleneck,
                               layers=[3, 4, 6, 3])
else:
    print('not implemented or in train.py')




dummy_inputs = torch.zeros((1, slices, channels, height, width))


with torch.no_grad():
    dummy_out = model.forward(dummy_inputs, dummy_inputs)
    best_model_dummy_out = best_model.forward(dummy_inputs, dummy_inputs)

model = model.to(device)
best_model = best_model.to(device)


if args.num_gpus_to_use > 1:
    model = nn.DataParallel(model)


if args.resume_from_baseline:
    baseline_filepath, _, _ = build_experiment_folder(
        experiment_name='baseline',
        log_path=args.logs_path + "/" + "epochs_" + str(args.max_epochs),

    )
    _ = restore_model(restore_fields = {"model": model}, path=baseline_filepath, device=device, best = True)





weights = [1-(len(class_0)/(len(class_0)+len(class_1))), len(class_0)/(len(class_0)+len(class_1))]

print('weights', weights)
criterion= nn.CrossEntropyLoss(weight = torch.FloatTensor(weights).to(device))


if args.optim.lower() == "sgd":
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
else:
    optimizer = optim.Adam(
        model.parameters(), lr=1e-05, weight_decay=args.weight_decay
    )

if args.scheduler == "CosineAnnealing":
    scheduler = CosineAnnealingLR(
        optimizer=optimizer, T_max=args.max_epochs, eta_min=args.lr_min
    )
else:
    scheduler = MultiStepLR(optimizer, milestones=int(args.max_epochs / 10), gamma=0.1)

############################################################################# Restoring

restore_fields = {
    "model": model,
    "optimizer": optimizer,
    "scheduler": scheduler,
}
#####################################
# log the logits, loss etc.
log_dict = {}
log_dict["logits"] = {}
log_dict["loss"] = {}
log_dict["MTL_loss"] = []
log_dict["gradients"] = {}

for name, param in model.named_parameters():
    log_dict["gradients"][name] = []

#######################################
start_epoch = 0


################################################################################# Metric
# track for each iteration with input of raw output from NN and targets
metrics_to_track = {
    "cross_entropy": lambda x, y: torch.nn.CrossEntropyLoss()(x, y).item(),
    "accuracy": compute_accuracy,
}


############################################################################## Training

def train_iter(metric_tracker, model, x,sal, y, ids, iteration, epoch, set_name):

    inputs = x.to(device)
    sal = sal.to(device)
    targets = torch.tensor(y, dtype=torch.long).to(device)
    model = model.train()
    logits = model(inputs, sal)

    # print(logits, targets)

    loss = criterion(
        input=logits,
        target=targets)


    metric_tracker.push(epoch, iteration, logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    log_string = "{}, epoch: {} {} iteration: {}; {}".format(
        args.experiment_name,
        epoch,
        set_name,
        iteration,
        "".join(
            "{}: {:0.4f}; ".format(key, value[-1])
            if key not in ["epochs", "iterations"] and len(value) > 0
            else ""
            for key, value in metric_tracker.metrics.items()
        ),
    )

    return log_string


def eval_iter(metric_tracker, model, x,sal, y, ids, iteration, epoch, set_name, final_pred, final_label):

    inputs = x.to(device)
    sal = sal.to(device)
    targets = torch.tensor(y, dtype=torch.long).to(device)

    with torch.no_grad():
        model = model.eval()
        logits = model(inputs, sal)


    metric_tracker.push(epoch, iteration, logits, targets)
    m = nn.Softmax(dim=-1)


    for i, el in enumerate(torch.argmax(m(logits), dim=1)):
        final_pred.append(int(torch.argmax(m(logits), dim=1)[i]))

    for i, el in enumerate(targets):
            final_label.append(int(targets[i]))


    log_string = "{}, epoch: {} {} iteration: {}; {}".format(
        args.experiment_name,
        epoch,
        set_name,
        iteration,
        "".join(
            "{}: {:0.4f}; ".format(key, value[-1])
            if key not in ["epochs", "iterations"] and len(value) > 0
            else ""
            for key, value in metric_tracker.metrics.items()
        ),
    )

    return log_string, final_pred, final_label


def run_epoch(epoch, model, training, data_loader, metric_tracker):
    iterations = epoch * (len(data_loader) )
    print(f"{len(data_loader) } batches.")

    final_pred = []
    final_label = []
    with tqdm.tqdm(initial=0, total=len(data_loader), smoothing=0) as pbar:
        # Clear this for every epoch. Only save the best epoch results.
        for x,sal, y, ids in data_loader:

            if training:
                log_string = train_iter(
                    model=model,
                    x=x,
                    sal = sal,
                    y = y,
                    ids = ids,
                    iteration=iterations,
                    epoch=epoch,
                    set_name=metric_tracker.tracker_name,
                    metric_tracker=metric_tracker
                )

            else:
                log_string, final_pred, final_label = eval_iter(
                    model=model,
                    x=x,
                    y=y,
                    sal = sal,
                    ids = ids,
                    iteration=iterations,
                    epoch=epoch,
                    set_name=metric_tracker.tracker_name,
                    metric_tracker=metric_tracker,
                    final_pred=final_pred,
                    final_label=final_label
                )

            pbar.set_description(log_string)
            pbar.update(1)
            iterations += 1
    return final_pred, final_label if not training else None


def save_model(saved_models_filepath, latest_epoch_filename, best_epoch_filename, is_best):
    if args.save:
        state = {
            "args": args,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }

        # save the latest epoch model
        epoch_pbar.set_description(
            "Saving latest checkpoint at {}/{}".format(
                saved_models_filepath, latest_epoch_filename
            )
        )
        save_checkpoint(
            state=state,
            directory=saved_models_filepath,
            filename=latest_epoch_filename,
            is_best=False,
        )
        # save the best model.
        best_model_path = ""
        if is_best:
            best_model_path = save_checkpoint(
                state=state,
                directory=saved_models_filepath,
                filename=best_epoch_filename,
                is_best=True,
            )
            epoch_pbar.set_description(
                "Saving best checkpoint at {}/{}".format(
                    saved_models_filepath, best_model_path
                )
            )
    return best_model_path


if __name__ == "__main__":
    loss_weight_dict = None


    if args.resume:
        start_epoch = restore_model(restore_fields, path=saved_models_filepath, device=device)

    metric_tracker_train = MetricTracker(
        metrics_to_track=metrics_to_track,
        load=True if start_epoch > 0 else False,
        path="",
        log_dict=log_dict,
        tracker_name="training"
    )
    metric_tracker_val = MetricTracker(
        metrics_to_track=metrics_to_track,
        load=True if start_epoch > 0 else False,
        path="",
        log_dict=log_dict,
        tracker_name="validation"
    )

    metric_tracker_test = MetricTracker(
        metrics_to_track=metrics_to_track,
        load=True if start_epoch > 0 else False,
        path="",
        log_dict=log_dict,
        tracker_name="testing"
    )

    with tqdm.tqdm(initial=start_epoch, total=args.max_epochs) as epoch_pbar:
        best_acc = 0.0
        best_epoch = 0
        each_epoch_acc = []


        for epoch in range(start_epoch, args.max_epochs):
            print()
            print(f"Running epoch {epoch}")
            best_epoch_stats = open(logs_filepath + "/best_epoch_stats.txt", "w")
            run_epoch(
                epoch,
                data_loader=train_loader,
                model=model,
                training=True,
                metric_tracker=metric_tracker_train,
            )
            final_predictions, final_labels = run_epoch(
                epoch,
                data_loader=val_loader,
                model=model,
                training=False,
                metric_tracker=metric_tracker_val,
            )

            scheduler.step()


            epoch_metric = metric_tracker_val.collect_per_epoch()
            print('epoch_metric', epoch_metric)
            is_best= False

            final_predictions = np.array(final_predictions)
            final_labels = np.array(final_labels)
            temp = (final_predictions == final_labels)



            current_val_acc = float(sum(temp) / len(final_predictions))

            print(f"\ncurrent_val_acc: {current_val_acc:.4f}")
            latest_epoch_filename = "latest_ckpt.pth.tar"
            best_epoch_filename = "ckpt.pth.tar"
            if current_val_acc >= best_acc:
                best_acc = current_val_acc
                is_best = True
                best_epoch = epoch
                best_model_save_path = save_model(saved_models_filepath=saved_models_filepath,
                                                  latest_epoch_filename=latest_epoch_filename,
                                                  best_epoch_filename=best_epoch_filename,
                                                  is_best=is_best,
                                                  )
            else:
                is_best = False
                save_model(saved_models_filepath=saved_models_filepath,
                           latest_epoch_filename=latest_epoch_filename,
                           best_epoch_filename=best_epoch_filename,
                           is_best=is_best,
                           )
            print(
                "\nAll tasks till epoch: {} best_acc: {:.4f} best_epoch: {:}".format(
                    epoch, best_acc, best_epoch
                )
            )
            epoch_pbar.set_description("")
            epoch_pbar.update(1)
            # Save important stats snapshot.
            # Track if best epoch changed, only compute best epoch stats if changed
            print()
            best_epoch_stats.write(f"Till epoch: {epoch}\n")
            print(f"Best_val_epoch: {best_epoch}")
            print(f"Best_val_acc: {best_acc:.4f}")
            best_epoch_stats.write(f"Best_val_epoch: {best_epoch}\n")
            best_epoch_stats.write(f"Best_val_acc: {best_acc:.4f}\n")


            each_epoch_acc.append(current_val_acc)

            metric_tracker_train.plot(
                path="{}/train/metrics.png".format(images_filepath)
            )
            metric_tracker_val.plot(path="{}/val/metrics.png".format(images_filepath))
            metric_tracker_train.path = "{}/metrics_train.pt".format(logs_filepath.replace(os.sep, "/"))
            metric_tracker_train.save()
            metric_tracker_val.path = "{}/metrics_val.pt".format(logs_filepath.replace(os.sep, "/"))
            metric_tracker_val.save()
            # Save train log_dict
            metric_tracker_train_log_dict_list = metric_tracker_train.log_dict
            metric_tracker_train_log_dict_path = os.path.join(logs_filepath, "metric_tracker_train_log_dict.pt")
            save_metrics_dict_in_pt(path=metric_tracker_train_log_dict_path,
                                    metrics_dict=metric_tracker_train_log_dict_list,
                                    overwrite=True)
            metric_tracker_val_log_dict_list = metric_tracker_val.log_dict
            metric_tracker_val_log_dict_path = os.path.join(logs_filepath, "metric_tracker_val_log_dict.pt")
            save_metrics_dict_in_pt(path=metric_tracker_val_log_dict_path,
                                    metrics_dict=metric_tracker_val_log_dict_list,
                                    overwrite=True)

    final_predictions, final_labels = run_epoch(
        epoch,
        data_loader=test_loader,
        model=model,
        training=False,
        metric_tracker=metric_tracker_test,
    )

    final_predictions = np.array(final_predictions)
    final_labels = np.array(final_labels)
    temp = (final_predictions == final_labels)

    final_test_acc = float(sum(temp) / len(final_predictions))

    print(f"\nfinal_test_acc: {final_test_acc:.4f}")

    main_end = datetime.now()
    dt_string = main_end.strftime("%d/%m/%Y %H:%M:%S")
    print("End main() date and time =", dt_string)

    main_execute_time = main_end - main_start
    print("main() execute time: {}".format(main_execute_time))
    sys.exit(0)


