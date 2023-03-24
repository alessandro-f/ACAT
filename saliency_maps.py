import os
import pprint
import numpy as np
from utils.arg_parsing import parse_args
from utils.gpu_selection_utils import set_gpu_visible_devices
from utils.storage import (
    build_experiment_folder,
    save_checkpoint,
    restore_model,
    restore_model_from_path, save_metrics_dict_in_pt,
)
from datetime import datetime

main_start = datetime.now()
dt_string = main_start.strftime("%d/%m/%Y %H:%M:%S")
print("Start main() date and time =", dt_string)

args = parse_args()
set_gpu_visible_devices(num_gpus_to_use=args.num_gpus_to_use)


from torch.autograd import grad
import random
from models.autoencoder_architectures import *
import torch.backends.cudnn as cudnn
import nibabel as nib
import tifffile as tif
from scipy import ndimage

import torchvision
from torchvision import transforms
from utils.custom_transforms import SliceSamplingUniform
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.model_architectures import *

################################################################################## Data

height = 256
width = 256
channels = 1
slices = 19


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



class CustomImageDataset(Dataset):
    def __init__(self, images, process, transform):
        self.image_list = images
        self.transform = transform
        self.process = process

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = self.process(img_path)
        image = torch.tensor(image)
        image = image.permute(2, 0, 1).unsqueeze(1)
        image = self.transform(image)
        image = self.transform(image)
        return image,img_path



dataset = CustomImageDataset(images = images, process = process_scan, transform = transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)


############################################################################ Determinism
# Seeding can be annoying in pytorch at the moment. Based on my experience,
# the below means of seeding allows for deterministic experimentation.
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

print("-----------------------------------")
pprint.pprint(args, indent=4)
print("-----------------------------------")

################################################################################## Model
model = resnet50_baseline(block=Bottleneck,
                          layers=[3, 4, 6, 3])

dummy_inputs = torch.zeros((1, slices, channels, height, width))


with torch.no_grad():
    dummy_out = model.forward(dummy_inputs)

model = model.to(device)
if args.num_gpus_to_use > 1:
    model = nn.DataParallel(model)

baseline_filepath, _, _ = build_experiment_folder(
    experiment_name='baseline',
    log_path=args.logs_path + "/" + "epochs_" + str(args.max_epochs),
)
_ = restore_model(restore_fields = {"model": model}, path=baseline_filepath, device=device, best = True)



mod = 'AE_COVID_no_bottleneck_6_12_16_2d'
enc_out = 512


ae = AE_no_bottleneck_6_12_16(batch_size= args.batch_size, input_height = 128, enc_type='my_resnet18', first_conv=False, maxpool1=False, enc_out_dim=enc_out, latent_dim=int(enc_out/2), lr=0.0001)

AE_path = f'models/autoenc_mod_{mod}'

checkpoint = torch.load(AE_path, map_location=lambda storage, loc: storage)

ae.load_state_dict(checkpoint['model'])

ae = ae.to(device)
if args.num_gpus_to_use > 1:
    ae = nn.DataParallel(ae)
    ae.module.encoder = nn.DataParallel(ae.module.encoder)
    ae.module.decoder = nn.DataParallel(ae.module.decoder)
########################################################################## Optimisation
ae.eval()

def to_tensor_grad(z, requires_grad=False):
    z = torch.Tensor(z).to(device)
    z.requires_grad=requires_grad
    return z

def to_numpy(z):
    return z.data.cpu().numpy()

alpha = 100
beta = 0.1

def compute_counterfactual(z, z0, targets, criterion_class = nn.CrossEntropyLoss(),  criterion_norm = nn.L1Loss()):
    for i in range(20):
        # print(i)
        z = to_tensor_grad(z, requires_grad=True)
        z_0 = to_tensor_grad(z0)
        logits = model((ae.module.decoder(z)).view(-1, slices, 1, height, width))
        saliency_loss = criterion_class(input=logits, target=targets)
        distance = criterion_norm(z, z_0)
        # print('loss', saliency_loss, 'distance',distance)

        loss = saliency_loss + alpha * distance

        dl_dz = grad(loss, z)[0]
        z = z - beta * dl_dz
        z = to_numpy(z)
        # print(z)
    return to_tensor_grad(z)

def compute_saliency(z, im2):
    shifted_image_1 = ae.module.decoder(z)
    im2 = im2.view(-1, slices, height, width)
    shifted_image_1 = shifted_image_1.view(-1, slices, height, width).to('cpu')
    shifted_image_1 = shifted_image_1.detach().cpu().numpy()
    dimage = np.abs(im2.cpu().numpy() - shifted_image_1)

    return dimage





saliency_root = os.path.join(args.dataset_root_folder, 'saliency_maps')

if not os.path.exists(saliency_root):
    os.makedirs(saliency_root)



for batch_idx, (inputs, ids) in enumerate(loader):

    brain = ids[0]

    name = os.path.basename(brain)
    inputs = inputs.to(device)
    model = model.eval()

    im1_enc = inputs.view(-1, 1, height, width).to(device)
    im2 = ae.module.decoder(ae.module.encoder(im1_enc)).detach().to('cpu')

    z = to_numpy(ae.module.encoder(im1_enc))
    z0 = z

    logits = model(inputs)[:,1]

    # positive counterfactual
    targets = torch.tensor([1], dtype=torch.long).to(device)
    z_out = compute_counterfactual(z.copy(), z0.copy(), targets)
    dimage1 = compute_saliency(z_out, im2.clone())

    # negative counterfactual
    targets = torch.tensor([0], dtype=torch.long).to(device)
    z_out = compute_counterfactual(z.copy(), z0.copy(), targets)
    dimage2 = compute_saliency(z_out, im2.clone())


    dimage1 *= (1.0 / dimage1.max())
    dimage2 *= (1.0 / dimage1.max())

    dimage = (dimage1+dimage2)/2


    tif.imwrite(os.path.join(saliency_root, f'{name}_AVG_gifmap.tif'), dimage,
                photometric='minisblack')
    print('image saved:', os.path.join(saliency_root, f'{name}_AVG_gifmap.tif'))



