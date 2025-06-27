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
        "load_function": lambda: models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1),
        "part1_definition": lambda model: nn.Sequential(*list(model.features.children())[:6]),
        "transform_resize": (224, 224),
        "output_filename_prefix": "alexnet_part1"
    },
    "InceptionV3": {
        "load_function": lambda: models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1),
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
        "output_filename_prefix": "inceptionv3_part1"
    },
    "MobileNetV2": {
        "load_function": lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
        "part1_definition": lambda model: nn.Sequential(*list(model.features.children())[:8]),
        "transform_resize": (224, 224),
        "output_filename_prefix": "mobilenetv2_part1"
    },
    "ResNet18": {
        "load_function": lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
        "part1_definition": lambda model: nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2
        ),
        "transform_resize": (224, 224),
        "output_filename_prefix": "resnet18_part1"
    },
    
    "VGG16": {
        "load_function": lambda: models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
        "part1_definition": lambda model: nn.Sequential(*list(model.features.children())[:16]),
        "transform_resize": (224, 224),
        "output_filename_prefix": "vgg16_part1"
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

def run_server_for_model(model_name, config, output_dir):
    """
    Runs the server logic for a specific model part and saves metrics.
    """
    print(f"\n[Server] --- Running for Model: {model_name} ---")

    # Load the full model and define its part1
    # Ensure model is on CPU initially as psutil checks are CPU-bound and pickling works directly.
    full_model = config["load_function"]()
    model_part1_instance = ModelPart(config["part1_definition"](full_model)).eval().cpu()

    # Data transformation and loading
    transform = transforms.Compose([
        transforms.Resize(config["transform_resize"]),
        transforms.ToTensor()
    ])
    dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=5, shuffle=False)

    # Socket setup
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"[Server - {model_name}] Waiting for client...")

    conn = None # Initialize conn
    try:
        conn, addr = s.accept()
        print(f"[Server - {model_name}] Connected to {addr}")

        server_metrics = [] # To store (infer_time, net_time, cpu, mem, throughput_kb_s)

        for i, (images, _) in enumerate(loader):
            batch_start = time.time()
            cpu = psutil.cpu_percent() # CPU usage since last call or boot
            mem = psutil.virtual_memory().percent
            t_infer_start = time.time()

            with torch.no_grad():
                output = model_part1_instance(images) # Inference on CPU

            infer_time = time.time() - t_infer_start
            
            # Ensure output is on CPU before pickling
            if output.is_cuda: # Should not be CUDA if model_part1_instance is .cpu()
                output = output.cpu()

            payload = {'tensor': output, 'start_time': batch_start, 'cpu': cpu, 'mem': mem, 'infer_time': infer_time}
            data_bytes = pickle.dumps(payload)
            size = len(data_bytes) # Size in bytes

            t_net_start = time.time()
            # Send payload size (4 bytes, big-endian)
            conn.sendall(size.to_bytes(4, 'big'))
            # Send payload data
            conn.sendall(data_bytes)
            net_time = time.time() - t_net_start

            # Calculate throughput (KB/s) based on inference time for that batch
            throughput_kb_s = (size / infer_time / 1024) if infer_time > 0 else 0

            server_metrics.append((infer_time, net_time, cpu, mem, throughput_kb_s))
            print(f"[Server - {model_name}] Batch {i+1} - Inference: {infer_time:.4f}s, Network: {net_time:.4f}s, CPU: {cpu}%, Mem: {mem}%, Throughput: {throughput_kb_s:.2f} KB/s")

            if i == 15: # Limit to 16 batches for demonstration
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

    # Aggregate and save metrics
    if server_metrics:
        sm_np = np.array(server_metrics)
        avg = np.mean(sm_np, axis=0)
        std = np.std(sm_np, axis=0)

        metric_labels = ["Inference Time (s)", "Network Time (s)", "CPU Utilization (%)", "Memory Utilization (%)", "Throughput (KB/s)"]
        
        results = {
            "model_name": model_name,
            "device": "cpu", # Assuming CPU since no .cuda() is used here for pi1
            "metrics": {}
        }

        print(f"\n[Server - {model_name}] === AVERAGES WITH STANDARD DEVIATION ===")
        for i, label in enumerate(metric_labels):
            print(f"{label}: {avg[i]:.4f} Â± {std[i]:.4f}")
            results["metrics"][label.replace(" ", "_").replace("(", "").replace(")", "")] = { # Clean labels for JSON keys
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
        # It's important that run_server_for_model handles socket setup and teardown for each run.
        run_server_for_model(model_name, config, args.output_dir)

    print("\n[Server] All specified model evaluations complete.")