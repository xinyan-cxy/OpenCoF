# Copyright (c) 2023 Zhongjie Duan and the ModelScope Community
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of DiffSynth-Studio, released under the Apache License 2.0:
# https://github.com/modelscope/DiffSynth-Studio
accelerate launch examples/flux/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata_attrictrl.csv \
  --data_file_keys "image" \
  --max_pixels 1048576 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/*.safetensors,black-forest-labs/FLUX.1-dev:ae.safetensors,DiffSynth-Studio/AttriCtrl-FLUX.1-Dev:models/brightness.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.value_controller.encoders.0." \
  --output_path "./models/train/FLUX.1-dev-AttriCtrl_full" \
  --trainable_models "value_controller" \
  --extra_inputs "value_controller_inputs" \
  --use_gradient_checkpointing
