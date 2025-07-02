import torch
import socket
import pickle
import time
import os
import json
import psutil
import numpy as np
import argparse
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn as nn

HOST = '0.0.0.0' # Listen on all available interfaces
PORT = 5555

# Define the models and their specific first parts, transformations, and output prefixes
MODELS_CONFIG = {
    "AlexNet": {
        "load_function": lambda: models.alexnet(weights=None),  # No pretrained weights
        "part1_definition": lambda model: nn.Sequential(*list(model.features.children())[:6]),
        "transform_resize": (224, 224),
        "output_filename_prefix": "alexnet_part1",
        "modify_classifier": lambda model: nn.Linear(4096, 10)  # CIFAR-10 has 10 classes
    },
    "InceptionV3": {
        "load_function": lambda: models.inception_v3(weights=None, aux_logits=False),
        "part1_definition": lambda model: nn.Sequential(
            model.Conv2d_1a_3x3,
            model.Conv2d_2a_3x3,
            model.Conv2d_2b_3x3,
            model.maxpool1,
            model.Conv2d_3b_1x1,
            model.Conv2d_4a_3x3,
            model.maxpool2,
            model.Mixed_5b,
            model.Mixed_5c
        ),
        "transform_resize": (299, 299),
        "output_filename_prefix": "inceptionv3_part1",
        "modify_classifier": lambda model: nn.Linear(2048, 10)
    },
    "MobileNetV2": {
        "load_function": lambda: models.mobilenet_v2(weights=None),
        "part1_definition": lambda model: nn.Sequential(*list(model.features.children())[:8]),
        "transform_resize": (224, 224),
        "output_filename_prefix": "mobilenetv2_part1",
        "modify_classifier": lambda model: nn.Linear(1280, 10)
    },
    "ResNet18": {
        "load_function": lambda: models.resnet18(weights=None),
        "part1_definition": lambda model: nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2
        ),
        "transform_resize": (224, 224),
        "output_filename_prefix": "resnet18_part1",
        "modify_classifier": lambda model: nn.Linear(512, 10)
    },
    "VGG16": {
        "load_function": lambda: models.vgg16(weights=None),
        "part1_definition": lambda model: nn.Sequential(*list(model.features.children())[:16]),
        "transform_resize": (224, 224),
        "output_filename_prefix": "vgg16_part1",
        "modify_classifier": lambda model: nn.Linear(4096, 10)
    }
}

class ModelPart(nn.Module):
    """
    A generic class to wrap a part of a model.
    """
    def __init__(self, part):
        super().__init__()
        self.part = part

    def forward(self, x):
        return self.part(x)

def load_cifar10_model(model_name, config):
    """Load model with CIFAR-10 weights if available, otherwise use ImageNet and modify."""
    # First try to load CIFAR-10 specific weights
    weight_files = {
        "MobileNetV2": "./models/mobilenetv2_cifar10.pth",
        "ResNet18": "./models/resnet18_cifar10.pth",
        "AlexNet": "./models/alexnet_cifar10.pth",
        "InceptionV3": "./models/inception_cifar10.pth",
        "VGG16": "./models/vgg16_cifar10.pth"
    }
    
    # Load base model
    full_model = config["load_function"]()
    
    # Modify classifier for CIFAR-10 (10 classes)
    if model_name == "MobileNetV2":
        full_model.classifier[1] = config["modify_classifier"](full_model)
    elif model_name in ["ResNet18", "InceptionV3"]:
        full_model.fc = config["modify_classifier"](full_model)
    elif model_name in ["AlexNet", "VGG16"]:
        full_model.classifier[-1] = config["modify_classifier"](full_model)
    
    # Try to load CIFAR-10 weights
    weight_file = weight_files.get(model_name)
    if weight_file and os.path.exists(weight_file):
        try:
            state_dict = torch.load(weight_file, map_location="cpu")
            full_model.load_state_dict(state_dict)
            print(f"[Server - {model_name}] Loaded CIFAR-10 trained weights from {weight_file}")
        except Exception as e:
            print(f"[Server - {model_name}] Failed to load CIFAR-10 weights: {e}")
            print(f"[Server - {model_name}] Using random initialization")
    else:
        print(f"[Server - {model_name}] No CIFAR-10 weights found, using random initialization")
    
    return full_model

def run_server_for_model(model_name, config, output_dir):
    """
    Runs the server logic for a specific model part and saves metrics.
    """
    print(f"\n[Server] --- Running for Model: {model_name} ---")

    # Load the model with CIFAR-10 configuration
    full_model = load_cifar10_model(model_name, config)
    model_part1_instance = ModelPart(config["part1_definition"](full_model)).eval().cpu()

    # Data transformation and loading
    transform = transforms.Compose([
        transforms.Resize(config["transform_resize"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 normalization
    ])
    dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=5, shuffle=False)

    # Socket setup
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"[Server - {model_name}] Waiting for client...")


    conn = None
    try:
        conn, addr = s.accept()
        print(f"[Server - {model_name}] Connected to {addr}")

        server_metrics = []
        # CORRECTED: All logic is now inside the loop to run per-batch.
        for i, (images, _) in enumerate(loader):
            batch_start_time = time.time()
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            
            # CORRECTED: Measure inference time for this specific batch.
            t_infer_start = time.time()
            with torch.no_grad():
                output = model_part1_instance(images)
            infer_time = time.time() - t_infer_start

            output = output.cpu()
            
            # CORRECTED: Use the correctly calculated 'infer_time' for the payload.
            payload = {'tensor': output, 'start_time': batch_start_time, 'cpu': cpu, 'mem': mem, 'infer_time': infer_time}
            data_bytes = pickle.dumps(payload)
            size = len(data_bytes)
            
            # CORRECTED: Measure network time for this specific transmission.
            t_net_start = time.time()
            conn.sendall(size.to_bytes(4, 'big'))
            conn.sendall(data_bytes)
            net_time = time.time() - t_net_start

            # CORRECTED: Append metrics for this batch.
            server_metrics.append((infer_time, net_time, cpu, mem))
            print(f"[Server - {model_name}] Batch {i+1} -> Inference: {infer_time:.4f}s, Network: {net_time:.4f}s, CPU: {cpu}%, Mem: {mem}%")

            if i == 7: # Limit to 16 batches for demonstration
                break

    except Exception as e:
        print(f"[Server - {model_name}] Error during communication or inference: {e}")
    finally:
        if conn:
            try:
                # Send a zero-length payload as an explicit end-of-data signal
                conn.sendall(b'\x00\x00\x00\x00')
                print(f"[Server - {model_name}] Sent end-of-data signal.")
            except Exception as e_send_close:
                print(f"[Server - {model_name}] Error sending close signal: {e_send_close}")
            conn.close()
            print(f"[Server - {model_name}] Connection to client closed.")
        s.close()
        print(f"[Server - {model_name}] Server socket closed.")

    # Aggregate and save metrics (This part was okay)
    if server_metrics:
        sm_np = np.array(server_metrics)
        avg = np.mean(sm_np, axis=0)
        std = np.std(sm_np, axis=0)

        metric_labels = ["Inference Time (s)", "Network Time (s)", "CPU Utilization (%)", "Memory Utilization (%)"]
        
        results = {
            "model_name": model_name,
            "device": "cpu",
            "metrics": {}
        }

        print(f"\n[Server - {model_name}] === AVERAGES WITH STANDARD DEVIATION ===")
        for i, label in enumerate(metric_labels):
            print(f"{label}: {avg[i]:.4f} ± {std[i]:.4f}")
            results["metrics"][label.replace(" ", "_").replace("(", "").replace(")", "")] = {
                "average": float(avg[i]),
                "std_dev": float(std[i])
            }
        
        output_filepath = os.path.join(output_dir, f"{config['output_filename_prefix']}_metrics.json")
        with open(output_filepath, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"[Server - {model_name}] Metrics saved to {output_filepath}")
    else:
        print(f"[Server - {model_name}] No metrics collected for {model_name}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run server-side inference for various PyTorch models and save metrics.")
    parser.add_argument("--output_dir", type=str, default="server_metrics",
                        help="Directory to save the JSON metric files.")
    parser.add_argument("--models", nargs='*', default=["all"],
                        help="List of models to run (e.g., AlexNet InceptionV3). Use 'all' to run all models.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    models_to_run = MODELS_CONFIG.keys()
    if "all" not in args.models:
        models_to_run = [m for m in args.models if m in MODELS_CONFIG]
        if not models_to_run:
            print("No valid models selected. Please choose from:", list(MODELS_CONFIG.keys()))
            exit()

    for model_name in models_to_run:
        config = MODELS_CONFIG[model_name]
        run_server_for_model(model_name, config, args.output_dir)

    print("\n[Server] All specified model evaluations complete.")
