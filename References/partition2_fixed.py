import torch
import socket
import pickle
import time
import os
import json
import psutil
import numpy as np
import argparse
from torchvision import models
import torch.nn as nn

# Default HOST - IMPORTANT: Change this to your server's IP if it's different
HOST = '10.100.117.4' 
PORT = 5555

# Define the models and their specific second parts
MODELS_CONFIG = {
    "AlexNet": {
        "load_function": lambda: models.alexnet(weights=None),
        "part2_definition": lambda model: nn.Sequential(
            *list(model.features.children())[6:], # Continue from where server left off
            model.avgpool,
            nn.Flatten(),
            model.classifier
        ),
        "output_filename_prefix": "alexnet_part2",
        "modify_classifier": lambda model: nn.Linear(4096, 10)
    },
    "InceptionV3": {
        "load_function": lambda: models.inception_v3(weights=None, aux_logits=False),
        "part2_definition": lambda model: nn.Sequential(
            model.Mixed_5d,
            model.Mixed_6a,
            model.Mixed_6b,
            model.Mixed_6c,
            model.Mixed_6d,
            model.Mixed_6e,
            model.Mixed_7a,
            model.Mixed_7b,
            model.Mixed_7c,
            model.avgpool,
            nn.Flatten(),
            model.dropout,
            model.fc
        ),
        "output_filename_prefix": "inceptionv3_part2",
        "modify_classifier": lambda model: nn.Linear(2048, 10)
    },
    "MobileNetV2": {
        "load_function": lambda: models.mobilenet_v2(weights=None),
        "part2_definition": lambda model: nn.Sequential(
            *list(model.features.children())[8:], # Continue from where server left off
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            model.classifier
        ),
        "output_filename_prefix": "mobilenetv2_part2",
        "modify_classifier": lambda model: nn.Linear(1280, 10)
    },
    "ResNet18": {
        "load_function": lambda: models.resnet18(weights=None),
        "part2_definition": lambda model: nn.Sequential(
            model.layer3, model.layer4, model.avgpool,
            nn.Flatten(), model.fc
        ),
        "output_filename_prefix": "resnet18_part2",
        "modify_classifier": lambda model: nn.Linear(512, 10)
    },
    "VGG16": {
        "load_function": lambda: models.vgg16(weights=None),
        "part2_definition": lambda model: nn.Sequential(
            *list(model.features.children())[16:], # Continue from where server left off
            model.avgpool,
            nn.Flatten(),
            model.classifier
        ),
        "output_filename_prefix": "vgg16_part2",
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
    """Load model with CIFAR-10 weights if available."""
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
            print(f"[Client - {model_name}] Loaded CIFAR-10 trained weights from {weight_file}")
        except Exception as e:
            print(f"[Client - {model_name}] Failed to load CIFAR-10 weights: {e}")
            print(f"[Client - {model_name}] Using random initialization")
    else:
        print(f"[Client - {model_name}] No CIFAR-10 weights found, using random initialization")
    
    return full_model

def recvall(sock, length):
    """
    Helper function to ensure all bytes are received from the socket.
    """
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise EOFError("Socket closed early or no more data to receive.")
        data += more
    return data

def run_client_for_model(model_name, config, server_host, output_dir):
    """
    Runs the client logic for a specific model part and saves metrics.
    """
    print(f"\n[Client] --- Running for Model: {model_name} ---")

    # Load the model with CIFAR-10 configuration
    full_model = load_cifar10_model(model_name, config)
    model_part2_instance = ModelPart(config["part2_definition"](full_model)).eval().cpu()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((server_host, PORT))
        print(f"[Client - {model_name}] Connected to server at {server_host}:{PORT}")
    except ConnectionRefusedError:
        print(f"[Client - {model_name}] Connection refused. Is the server running at {server_host}:{PORT}?")
        return
    except Exception as e:
        print(f"[Client - {model_name}] Error connecting to server: {e}")
        return

    combined_metrics = [] # To store (infer1, infer2, cpu1, cpu2, mem1, mem2, total_e2e, throughput, accuracy)
    total_correct = 0
    total_samples = 0

    try:
        batch_count = 0
        while True: # Loop indefinitely until end-of-data signal or error
            # Receive size of payload
            size_bytes = recvall(s, 4)
            size = int.from_bytes(size_bytes, 'big')

            if size == 0: # Explicit end-of-data signal from server
                print(f"[Client - {model_name}] Received end-of-data signal from server.")
                break # Exit the loop

            # Receive payload data
            data = recvall(s, size)
            payload = pickle.loads(data)

            tensor = payload['tensor']
            labels = payload['labels']  # GET LABELS FOR ACCURACY
            batch_start_time = payload['start_time']
            cpu1, mem1, infer1 = payload['cpu'], payload['mem'], payload['infer_time']

            cpu2 = psutil.cpu_percent() # CPU usage on client since last call
            mem2 = psutil.virtual_memory().percent # Memory usage on client
            t_infer_start = time.time()

            with torch.no_grad():
                out = model_part2_instance(tensor) # Perform inference on client

            infer2 = time.time() - t_infer_start
            
            # Calculate accuracy
            _, predicted = torch.max(out.data, 1)
            batch_correct = (predicted == labels).sum().item()
            total_correct += batch_correct
            total_samples += labels.size(0)
            batch_accuracy = (batch_correct / labels.size(0)) * 100.0
            
            # Calculate throughput (items/second or batch_size/infer_time_pi2)
            throughput = (tensor.size(0) / infer2) if infer2 > 0 else 0 

            total_end_to_end = time.time() - batch_start_time # Total time for this batch

            combined_metrics.append((infer1, infer2, cpu1, cpu2, mem1, mem2, total_end_to_end, throughput, batch_accuracy))
            batch_count += 1

            print(f"[Client - {model_name}] Batch {batch_count} - Pi1 Infer: {infer1:.4f}s, Pi2 Infer: {infer2:.4f}s, "
                  f"Total E2E: {total_end_to_end:.4f}s, Throughput: {throughput:.4f} samples/s, Accuracy: {batch_accuracy:.2f}%")
            
            if batch_count >= 16: # Client-side limit to match server's 16 batches
                print(f"[Client - {model_name}] Processed {batch_count} batches, expecting server to send end signal now.")

    except EOFError:
        print(f"[Client - {model_name}] Server closed connection unexpectedly or sent incomplete data stream.")
    except Exception as e:
        print(f"[Client - {model_name}] Error during data reception or inference: {e}")
    finally:
        s.close()
        print(f"[Client - {model_name}] Connection closed.")

    # Calculate overall accuracy
    overall_accuracy = (total_correct / total_samples * 100.0) if total_samples > 0 else 0.0
    print(f"[Client - {model_name}] Overall accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_samples})")

    # Aggregate and save metrics
    if combined_metrics:
        cm_np = np.array(combined_metrics)
        avg = np.mean(cm_np, axis=0)
        std = np.std(cm_np, axis=0)

        metric_labels = [
            "Pi1 Inference Time (s)", "Pi2 Inference Time (s)",
            "Pi1 CPU Utilization (%)", "Pi2 CPU Utilization (%)",
            "Pi1 Memory Utilization (%)", "Pi2 Memory Utilization (%)",
            "Total End-to-End Time (s)", "Throughput (samples/s)",
            "Batch Accuracy (%)"
        ]
        
        results = {
            "model_name": model_name,
            "device": "cpu",
            "server_host": server_host,
            "overall_accuracy": overall_accuracy,
            "total_samples": total_samples,
            "metrics": {}
        }

        print(f"\n[Client - {model_name}] === AVERAGES WITH STANDARD DEVIATION ===")
        for i, label in enumerate(metric_labels):
            print(f"{label}: {avg[i]:.4f} +/- {std[i]:.4f}")
            results["metrics"][label.replace(" ", "_").replace("(", "").replace(")", "")] = {
                "average": float(avg[i]),
                "std_dev": float(std[i])
            }
        
        output_filepath = os.path.join(output_dir, f"{config['output_filename_prefix']}_metrics.json")
        with open(output_filepath, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"[Client - {model_name}] Metrics saved to {output_filepath}")
    else:
        print(f"[Client - {model_name}] No metrics collected for {model_name}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run client-side inference for various PyTorch models and save metrics.")
    parser.add_argument("--output_dir", type=str, default="client_metrics",
                        help="Directory to save the JSON metric files.")
    parser.add_argument("--models", nargs='*', default=["all"],
                        help="List of models to run (e.g., AlexNet InceptionV3). Use 'all' to run all models.")
    parser.add_argument("--host", type=str, default=HOST,
                        help=f"The IP address of the server (default: {HOST}).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    models_to_run = MODELS_CONFIG.keys()
    if "all" not in args.models:
        models_to_run = [m for m in args.models if m in MODELS_CONFIG]
        if not models_to_run:
            print("No valid models selected. Please choose from:", list(MODELS_CONFIG.keys()))
            exit()

    print(f"Client will attempt to connect to server at {args.host}:{PORT}")
    print("Please ensure the server (partition1_fixed.py) is running and configured correctly.")

    for model_name in models_to_run:
        config = MODELS_CONFIG[model_name]
        # Run client for each model sequentially
        run_client_for_model(model_name, config, args.host, args.output_dir)

    print("\n[Client] All specified model evaluations complete.")