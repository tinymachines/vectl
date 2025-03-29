#!/bin/bash

# Device preparation script
# Usage: ./prepare_device.sh /dev/sdX

set -e  # Exit on error

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /dev/sdX"
    exit 1
fi

DEVICE="$1"
./reset_device.sh "${DEVICE}"
echo "===== Device Preparation Script ====="
echo "Target device: $DEVICE"

# Check if device exists
if [ ! -b "$DEVICE" ]; then
    echo "Error: Device $DEVICE does not exist or is not a block device."
    exit 1
fi

# Unmount any mounted partitions from this device
echo "Unmounting any mounted partitions..."
mount | grep "$DEVICE" | awk '{print $1}' | while read partition; do
    echo "Unmounting $partition..."
    umount "$partition" || true
done

# Wipe existing partitions
echo "Wiping existing partitions..."
wipefs --all "$DEVICE"

# Display device information
echo "Device information after wiping:"
fdisk -l "$DEVICE"

echo "Device preparation completed successfully."
exit 0
