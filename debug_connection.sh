#!/bin/bash
# Debug script to test basic connectivity

echo "=== Debugging RPC Connection Issues ==="
echo ""

# Clean up any existing processes
echo "Cleaning up existing processes..."
ssh master-pi "pkill -f distributed_runner" 2>/dev/null || true
ssh core-pi "pkill -f distributed_runner" 2>/dev/null || true
pkill -f distributed_runner 2>/dev/null || true

echo ""
echo "Testing network connectivity..."
echo "Pinging master-pi..."
ping -c 2 master-pi || echo "Failed to ping master-pi"

echo ""
echo "Pinging core-pi..."
ping -c 2 core-pi || echo "Failed to ping core-pi"

echo ""
echo "Checking if port 44444 is available on this machine..."
netstat -tuln | grep 44444 || echo "Port 44444 is not in use"

echo ""
echo "Let me know what you find and we can proceed with debugging!"