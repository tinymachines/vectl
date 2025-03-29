#!/bin/bash
if [ $# -ne 1 ]; then
  echo "Usage: $0 /dev/sdX"
  exit 1
fi

DEVICE=$1

echo "Resetting device $DEVICE..."
echo "WARNING: All data on $DEVICE will be lost!"
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Operation cancelled."
  exit 1
fi

# Unmount any mounted partitions
umount ${DEVICE}* 2>/dev/null

# Wipe the first 100MB of the device (clears any existing header)
dd if=/dev/zero of=$DEVICE bs=1M count=100 status=progress
sync

echo "Device reset complete!"
