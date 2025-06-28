#!/bin/bash
# Script to safely backup and reorganize model files on Pis

echo "=== Backing up and reorganizing model files on Pis ==="

# Function to process each Pi
process_pi() {
    local pi_name=$1
    echo ""
    echo "Processing $pi_name..."
    
    # Create backup
    echo "  1. Creating backup of existing models file..."
    ssh $pi_name "cd ~/pytorch && cp models models.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Create models directory
    echo "  2. Creating models directory..."
    ssh $pi_name "cd ~/pytorch && mkdir -p models_dir"
    
    # Extract the model to a temporary location
    echo "  3. Extracting model from zip..."
    ssh $pi_name "cd ~/pytorch && unzip -q models -d models_dir/"
    
    # Move the extracted model to proper location
    echo "  4. Reorganizing extracted files..."
    ssh $pi_name "cd ~/pytorch && mv models_dir/mobilenetv2_cifar10 models_dir/mobilenetv2_cifar10.pth"
    
    # Remove the old models file and rename directory
    echo "  5. Cleaning up..."
    ssh $pi_name "cd ~/pytorch && rm models && mv models_dir models"
    
    # Verify
    echo "  6. Verifying..."
    ssh $pi_name "cd ~/pytorch && ls -la models/"
}

# Process both Pis
process_pi "worker1-pi"
process_pi "worker2-pi"

echo ""
echo "=== Reorganization Complete ==="
echo ""
echo "Original zip files backed up as: models.backup.<timestamp>"
echo "Model files now in: ~/pytorch/models/mobilenetv2_cifar10.pth"
echo ""
echo "To test the model loading:"
echo "ssh <pi> \"cd ~/pytorch && python3 -c 'import torch; m = torch.load(\\\"models/mobilenetv2_cifar10.pth\\\"); print(\\\"Model loaded successfully\\\")'\"" 