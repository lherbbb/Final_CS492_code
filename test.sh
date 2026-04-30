#!/bin/bash


python test.py --model="LCM" --output_dir="output/" --num_validation_set=1000  --lora_input_dir="lcm-adversarial-constant-loss-bound-v2/checkpoint-20000"  --strength=0.2 --num_inference_step=5 --device="cuda:2"
python test.py --model="LCM" --output_dir="output/" --num_validation_set=1000  --lora_input_dir="lcm-adversarial-constant-loss-bound-v2/checkpoint-20000"  --strength=0.4 --num_inference_step=4 --device="cuda:2"
