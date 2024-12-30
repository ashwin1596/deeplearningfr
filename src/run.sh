export DEVICE_TYPE=GPU
# export DEVICE_TYPE=CPU

export CUDA_VISIBLE_DEVICES=0

# export BATCH_SIZE=1
# export BATCH_SIZE=32
export BATCH_SIZE=1024
# export BATCH_SIZE=2048

export NUM_EPOCHS=5

# compute-sanitizer ./frame
# cuda-gdb ./frame
./frame
# ./infer