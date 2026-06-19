# !/bin/bash

C_APP_ROOT=/app
C_APP_DEPS="ldconfig && nvidia-smi -L && nvidia-debugdump -l && cd /app && export PYTHONPATH=\$(pwd)/build:\$PYTHONPATH"
# TODO: Figure out why using --prefix to install in a directory worked previously, but, fails now.
# -it jonathanporta/tf2:2.3.1-gpu bash -c "${C_APP_DEPS} && pip install --prefix \$(pwd)/build -r requirements.txt && python ${1}"

# Parameter Explanation:
#   -u 0:$(id -g) \ # Run container as root:local_user so that any output files won't be totally unreadable without a chown
#   -e TF_GPU_THREAD_MODE=gpu_private \ # https://www.tensorflow.org/guide/gpu_performance_analysis
#  NVIDIA NCCL Debug Flags - For info on why we need these: https://github.com/NVIDIA/nccl/issues/342
#  NVIDIA NCCL Flag Docs: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
#   -e NCCL_DEBUG=WARN \
#   -e NCCL_SHM_DISABLE=1 \
#
#  map local keras datasets directory into container to avoid extra download time
#   -v $HOME/.keras:/root/.keras \

# Launch docker container for a python training session
echo "Will run '${1}' via Python in a Tensorflow GPU Supported Container!"
docker run --rm \
  -u 0:$(id -g) \
  --privileged \
  --network host \
  --gpus all \
  --shm-size=1024m \
  -e NVIDIA_VISIBLE_DEVICES="all" \
  -e NVIDIA_DRIVER_CAPABILITIES="all" \
  -e NCCL_DEBUG=INFO \
  -e NCCL_SHM_DISABLE=0 \
  -e TF_CPP_MIN_VLOG_LEVEL=0 \
  -e TF_CPP_MIN_LOG_LEVEL=1 \
  -e TF_GPU_THREAD_MODE="gpu_private" \
  -v "$(pwd)":${C_APP_ROOT} \
  -v $HOME/.keras:/root/.keras \
  -it jonathanporta/tf2:2.4.0-gpu bash -c "${C_APP_DEPS} && pip install -r requirements.txt && python ${1}"


# Launch docker container shell for debugging environment
# echo "Will run 'bash' in a Tensorflow GPU Supported Container!"
# docker run --rm \
#   -u 0:$(id -g) \
#   --privileged \
#   --network host \
#   --gpus all \
#   --shm-size=1024m \
#   -e NVIDIA_VISIBLE_DEVICES="all" \
#   -e NVIDIA_DRIVER_CAPABILITIES="all" \
#   -e NCCL_DEBUG=INFO \
#   -e NCCL_SHM_DISABLE=0 \
#   -e TF_CPP_MIN_VLOG_LEVEL=0 \
#   -e TF_CPP_MIN_LOG_LEVEL=0 \
#   -e TF_GPU_THREAD_MODE="gpu_private" \
#   -v "$(pwd)":${C_APP_ROOT} \
#   -v $HOME/.keras:/root/.keras \
#   -it jonathanporta/tf2:2.4.0-gpu bash -c "${C_APP_DEPS} && bash"
