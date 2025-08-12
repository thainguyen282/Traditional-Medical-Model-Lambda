# VeneraMedicalM1

Clone Huggingface repository VeneraMEdicalM1-dataset
Update  .env file accord to .env.exapmle

To run with test experiemnt
```bash
python main.py --test
```



Run with deepspeed:
```bash
deepspeed --num_gpus=2 \
	main.py --config configs/experiments/unsloth-qwen3-4B-full/exp0001/config.yaml \
	--deepspeed
```
# Problems
- The launch_sweep.sh contains `/r` symbol, which is not accepted on Unix. Use _dos2unix_ to convert it.
- Run ```wandb init``` to set which entity (kinda team name) and project