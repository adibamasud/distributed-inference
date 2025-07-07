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
        # AlexNet features have 13 layers. Starts from where server left off (:10)
        "part2_definition": lambda model: nn.Sequential(
            *list(model.features.children())[10:], # Continue from server's new split point
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
            model.Mixed_6c, # Starts from where server left off
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
        # MobileNetV2 features have 19 layers. Starts from where server left off (:14)
        "part2_definition": lambda model: nn.Sequential(
            *list(model.features.children())[14:], # Continue from server's new split point
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
            model.layer4, # Starts from where server left off
            model.avgpool,
            nn.Flatten(),
            model.fc
        ),
        "output_filename_prefix": "resnet18_part2",
        "modify_classifier": lambda model: nn.Linear(512, 10)
    },
    "VGG16": {
        "load_function": lambda: models.vgg16(weights=None),
        # VGG16 features have 31 layers. Starts from where server left off (:22)
        "part2_definition": lambda model: nn.Sequential(
            *list(model.features.children())[22:], # Continue from server's new split point
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
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise EOFError("Socket closed early or no more data to receive.")
        data += more
    return data

def run_client_for_model(model_name, config, server_host, output_dir):
    print(f"\n[Client] --- Running for Model: {model_name} ---")
    
    # Load the full model to properly define the second part based on its architecture
    full_model = load_cifar10_model(model_name, config)
    model_part2_instance = ModelPart(config["part2_definition"](full_model)).eval().cpu()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((server_host, PORT))
        print(f"[Client - {model_name}] Connected to server at {server_host}:{PORT}")
    except Exception as e:
        print(f"[Client - {model_name}] Connection error: {e}")
        return

    # Store all individual batch metrics for saving
    all_batch_metrics_raw = []
    # To store the batch_start_time for system-wide throughput calculation
    inter_batch_start_times = []
    first_batch_size = None # To capture the batch size for throughput calculation

    try:
        batch_count = 0

        while True:
            size_bytes = recvall(s, 4)
            size = int.from_bytes(size_bytes, 'big')
            if size == 0:
                print(f"[Client - {model_name}] Received end-of-data signal from server.")
                break

            try:
                data_receive_start_time = time.time()
                data = recvall(s, size)
                data_receive_time = time.time() - data_receive_start_time

                payload = pickle.loads(data)
            except Exception as e:
                print(f"[Client - {model_name}] Failed to receive/parse payload: {e}")
                continue

            try:
                tensor = payload['tensor']
                current_batch_start_time = payload['start_time'] # This is tA in the diagram for the current batch
                cpu1, mem1, infer1 = payload['cpu'], payload['mem'], payload['infer_time']

                # Capture the batch size from the first received tensor
                if first_batch_size is None:
                    first_batch_size = tensor.size(0) 

                cpu2 = psutil.cpu_percent()
                mem2 = psutil.virtual_memory().percent

                with torch.no_grad():
                    infer2_start = time.time()
                    out = model_part2_instance(tensor)
                    infer2 = time.time() - infer2_start

                # Record the start time of this batch for inter-batch time calculation
                inter_batch_start_times.append(time.time())

                batch_infer2_throughput = (tensor.size(0) / infer2) if infer2 > 0 else 0
                total_end_to_end = time.time() - current_batch_start_time

                # Client-side network throughput calculation
                # 'size' variable holds the intermediate data size for this batch
                network_throughput_client = (size / data_receive_time) if data_receive_time > 0 else 0

                # Store individual batch metrics in a dictionary for clarity
                current_batch_data = {
                    "batch_number": batch_count + 1,
                    "pi1_inference_time_s": infer1,
                    "pi2_inference_time_s": infer2,
                    "pi1_cpu_utilization_percent": cpu1,
                    "pi2_cpu_utilization_percent": cpu2,
                    "pi1_memory_utilization_percent": mem1,
                    "pi2_memory_utilization_percent": mem2,
                    "intermediate_data_size_bytes": size, # The actual size of the pickled data
                    "network_throughput_bytes_per_s": network_throughput_client, # Client-side calculation
                    "total_end_to_end_time_s": total_end_to_end,
                    "current_pi2_throughput_samples_per_s": batch_infer2_throughput,
                    "batch_start_time": current_batch_start_time # For debugging/validation of inter-batch times
                }
                all_batch_metrics_raw.append(current_batch_data)
                
                batch_count += 1

                print(f"[Client - {model_name}] Batch {batch_count} - Pi1 Infer: {infer1:.4f}s, Pi2 Infer: {infer2:.4f}s, Data Size: {size}B, Net Thpt: {network_throughput_client:.2f}B/s, End-to-End: {total_end_to_end:.4f}s, Current Pi2 Throughput: {batch_infer2_throughput:.2f} samples/s")

                if batch_count >= 8: # Limit to 8 batches for demonstration
                    print(f"[Client - {model_name}] Processed {batch_count} batches.")
            except Exception as e:
                print(f"[Client - {model_name}] Error during inference: {e}")
                continue

    except EOFError:
        print(f"[Client - {model_name}] Server closed connection unexpectedly.")
    except Exception as e:
        print(f"[Client - {model_name}] Unexpected error: {e}")
    finally:
        s.close()
        print(f"[Client - {model_name}] Connection closed.")

    if all_batch_metrics_raw:
        # Prepare the metrics for averaging and standard deviation calculation
        data_for_numpy = []
        for batch_data in all_batch_metrics_raw:
            data_for_numpy.append([
                batch_data["pi1_inference_time_s"],
                batch_data["pi2_inference_time_s"],
                batch_data["pi1_cpu_utilization_percent"],
                batch_data["pi2_cpu_utilization_percent"],
                batch_data["pi1_memory_utilization_percent"],
                batch_data["pi2_memory_utilization_percent"],
                batch_data["intermediate_data_size_bytes"],
                batch_data["network_throughput_bytes_per_s"],
                batch_data["total_end_to_end_time_s"],
                batch_data["current_pi2_throughput_samples_per_s"]
            ])
        
        cm_np = np.array(data_for_numpy)
        avg = np.mean(cm_np, axis=0)
        std = np.std(cm_np, axis=0)

        # --- Recalculating Throughput based on Image Definition ---
        calculated_inter_batch_times = []
        if len(inter_batch_start_times) > 1:
            for i in range(1, len(inter_batch_start_times)):
                calculated_inter_batch_times.append(inter_batch_start_times[i] - inter_batch_start_times[i-1])

        image_inference_throughput = 0.0
        if calculated_inter_batch_times and first_batch_size is not None:
            average_inter_batch_time = np.mean(calculated_inter_batch_times)
            if average_inter_batch_time > 0:
                batch_throughput = 1 / average_inter_batch_time
                image_inference_throughput = first_batch_size * batch_throughput
            else:
                print(f"[Client - {model_name}] Warning: Average inter-batch time is zero, cannot calculate image inference throughput.")
        else:
            print(f"[Client - {model_name}] Warning: Not enough batch data to calculate image inference throughput or batch size unknown.")

        metric_labels = [
            "Pi1 Inference Time (s)", "Pi2 Inference Time (s)",
            "Pi1 CPU Utilization (%)", "Pi2 CPU Utilization (%)",
            "Pi1 Memory Utilization (%)", "Pi2 Memory Utilization (%)",
            "Intermediate Data Size (bytes)", "Network Throughput (B/s)", # New labels
            "Total End-to-End Time (s)", "Current Pi2 Throughput (samples/s)"
        ]

        results = {
            "model_name": model_name,
            "device": "cpu",
            "server_host": server_host,
            "overall_metrics_summary": {}, # Store averages and std devs here
            "individual_batch_metrics": all_batch_metrics_raw, # Store all raw batch data here
            "calculated_image_inference_throughput": {
                "average": float(image_inference_throughput),
                "std_dev": 0.0 # This is a single computed value, std_dev not applicable in the same context
            }
        }

        print(f"\n[Client - {model_name}] === OVERALL AVERAGES WITH STANDARD DEVIATION ===")
        for i, label in enumerate(metric_labels):
            print(f"{label}: {avg[i]:.4f} +/- {std[i]:.4f}")
            results["overall_metrics_summary"][label.replace(" ", "_").replace("(", "").replace(")", "")] = {
                "average": float(avg[i]),
                "std_dev": float(std[i])
            }
        
        print(f"Image Inference Throughput (overall): {image_inference_throughput:.4f} samples/s")


        output_filepath = os.path.join(output_dir, f"{config['output_filename_prefix']}_metrics.json")
        with open(output_filepath, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"[Client - {model_name}] Metrics saved to {output_filepath}")
    else:
        print(f"[Client - {model_name}] No metrics collected for {model_name}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run client-side inference for PyTorch models and save metrics.")
    parser.add_argument("--output_dir", type=str, default="client_metrics")
    parser.add_argument("--models", nargs='*', default=["all"])
    parser.add_argument("--host", type=str, default=HOST)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    models_to_run = MODELS_CONFIG.keys() if "all" in args.models else [m for m in args.models if m in MODELS_CONFIG]
    if not models_to_run:
        print("No valid models selected. Available:", list(MODELS_CONFIG.keys()))
        exit()

    print(f"Client will attempt to connect to server at {args.host}:{PORT}")
    print("Ensure the server is running.")

    for model_name in models_to_run:
        config = MODELS_CONFIG[model_name]
        run_client_for_model(model_name, config, args.host, args.output_dir)

    print("\n[Client] All specified model evaluations complete.")
