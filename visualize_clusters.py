from torchvision import datasets, transforms
import argparse
import torch
import vision_transformer as vits
from torchvision import models as torchvision_models
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

def extract_features(model, x, arch):
    if "resnet" in arch:
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)
        output = torch.flatten(x, 1)
        
    else:
        output = model(x)

    return output

def visualize_clusters(args):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    dataset = datasets.CIFAR10(root='/workspace/datasets', train=False, download=True, transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        num_workers=4,
        drop_last=False,
        shuffle=True
    )

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")
    
    # load the weights
    student = student.cuda()
    student.load_state_dict(torch.load(args.weight_path)['student'], strict=False)
    student.eval()

    feature_nps = []
    label_nps = []

    # visualize clusters in feature space
    cnt = 0
    num_batches = 100
    for i, (images, labels) in enumerate(data_loader):
        if i == num_batches:
            break

        images = images.cuda()
        with torch.no_grad():
            student_output = extract_features(student, images, args.arch)

            output_np = student_output.to('cpu').detach().numpy().copy()
            labels_np = labels.to('cpu').detach().numpy().copy()

            feature_nps.append(output_np)
            label_nps.append(labels_np)

    features_np = np.concatenate(feature_nps)
    labels_np = np.concatenate(label_nps)

    print(f"features_np.shape: {features_np.shape}")
    print(f"labels_np.shape: {labels_np.shape}")

    # visualize clusters in feature space

    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(features_np)

    df = pd.DataFrame(data={
        "x": features_2d[:, 0],
        "y": features_2d[:, 1],
        "label": labels_np.tolist()
    })
    df.plot(x="x", y="y", xlabel="x", ylabel="y", kind="scatter", c="label", colormap="tab10")
    plt.savefig(f'{args.output_dir}/vis.png')





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--weight_path', type=str, default='weights/weights.pth', help='path to the weights file')
    parser.add_argument('--output_dir', type=str, default='output', help='path to the output directory')
    parser.add_argument('--arch', default='vit_small', type=str,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    
    args = parser.parse_args()   
    visualize_clusters(args)
