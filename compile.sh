#!/bin/bash

# Step 1: Go to build directory
mkdir -p build
cd build || { echo "Failed to enter build directory"; exit 1; }

# Step 2: Ask user for build type
echo "Compile as Debug (default) or Release (1)?"
read -r choice

# Choose build type
if [ "$choice" == "1" ]; then
    BUILD_TYPE=Release
else
    BUILD_TYPE=Debug
fi

echo "Building $BUILD_TYPE..."

# Configure CMake
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ../src

# Step 3: Compile
make -j$(nproc)

# Step 4: Go back to parent directory
cd ..

# Step 5: Submit job
sbatch run.sh

