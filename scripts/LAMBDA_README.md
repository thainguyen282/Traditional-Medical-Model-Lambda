# Lambda Cloud training runner

Train CPT / SFT jobs on a Lambda Cloud GPU instance, push artifacts to HF Hub,
then auto-terminate the instance.

## Files — 3 script, 2 nơi chạy

| File | Chạy ở đâu | Vai trò |
|---|---|---|
| [lambda_launch.py](lambda_launch.py) | **máy local của bạn** | Tổng chỉ huy từ xa. Gọi Lambda API tạo instance, scp 2 script kia lên, khởi chạy, theo dõi log, cuối cùng gọi API terminate. Không bao giờ chạy trên instance. |
| [lambda_bootstrap.sh](lambda_bootstrap.sh) | **trên instance Lambda** | Script khởi động bên trong GPU instance. `apt install`, `git clone`, `pip install`, login HF/W&B, chạy `CPT.py` hay `main.py` tuỳ `TASK_TYPE`, gọi `lambda_push_to_hf.py`, viết sentinel `DONE_SUCCESS`/`DONE_FAILED` cho `launch.py` biết. |
| [lambda_push_to_hf.py](lambda_push_to_hf.py) | **trên instance Lambda** | Công cụ upload HF. `bootstrap.sh` gọi sau khi training xong. Đọc YAML để xác định artifact là CPT/CPT-SFT/SFT, tạo repo + branch, upload LoRA adapter + tokenizer. |

### Flow tổng quan

```
Máy bạn                          Lambda Cloud instance
────────                         ─────────────────────
lambda_launch.py ──API──▶ [tạo instance]
       │
       ├── scp ─▶  lambda_bootstrap.sh
       ├── scp ─▶  lambda_push_to_hf.py
       │
       └── ssh ─▶  bash lambda_bootstrap.sh
                         │
                         ├─ git clone, pip install
                         ├─ python CPT.py  (hoặc main.py)
                         └─ python lambda_push_to_hf.py ──▶ HF Hub

lambda_launch.py ──API──▶ [terminate instance]
```

### Tại sao phải tách 3 file thay vì gộp 1

- `launch.py` buộc phải ở **local** (chỉ nó biết IP/SSH key để kết nối instance).
- `bootstrap.sh` là **Bash** vì cần `apt install`, `source venv`, `tmux` — tự nhiên hơn Python.
- `push_to_hf.py` tách riêng vì logic đọc YAML + gọi `HfApi` dễ viết bằng Python, **và có thể chạy lại độc lập nếu push fail** (ssh vào, chạy tay với `--revision` khác).

## One-time setup

1. Register your **SSH public key** in the Lambda Cloud dashboard (Settings → SSH keys). Remember the name you gave it.
2. Create a **Lambda API key** (Settings → API keys).
3. Create a **write-enabled Hugging Face token** (https://huggingface.co/settings/tokens).
4. If the GitHub repo is private, create a **GitHub PAT** with `repo` scope.
5. Fill in `.env` (see [.env.example](../.env.example)):
   ```dotenv
   LAMBDA_API_KEY=secret_...
   LAMBDA_SSH_KEY_NAME=my-laptop-key
   LAMBDA_SSH_PRIVATE_KEY=~/.ssh/id_ed25519
   HF_TOKEN=hf_...
   WANDB_KEY=...            # optional
   GH_TOKEN=ghp_...         # only for private repos
   ```
6. Install local deps:
   ```bash
   pip install requests python-dotenv
   ```

## Usage

### Check available instance types
```bash
python scripts/lambda_launch.py --task sft --list-types
```

### Run a single SFT experiment
```bash
python scripts/lambda_launch.py \
    --task sft \
    --config configs/experiments/unsloth-qwen3-4B/exp0001/config.yaml
```
- `main.py` decides CPT-SFT vs pure SFT based on `config.model.name` (substring
  `cpt` → CPT-SFT).
- The HF revision (branch) will be
  `cpt-sft-<model-dir>-<exp-dir>-<job-tag>` or `sft-<model-dir>-<exp-dir>-<job-tag>`
  under `huyennkai/Venera-Traditional-Model`.

### Run CPT
```bash
python scripts/lambda_launch.py --task cpt
```
Pushes to `huyennkai/Venera-CPT-TMModel` on branch `cpt-<job-tag>`.

### Run multiple SFT configs on the same instance (cheaper)
Create a text file with one config per line:
```txt
# scripts/lambda_batch.txt
configs/experiments/unsloth-qwen3-4B/exp0001/config.yaml
configs/experiments/unsloth-qwen3-4B/exp0002/config.yaml
```
Then:
```bash
python scripts/lambda_launch.py --task sft --config-list scripts/lambda_batch.txt
```

### Debug / inspect after finish
```bash
python scripts/lambda_launch.py --task sft --config ... --keep-alive
# After the run, SSH in manually with the printed IP to inspect files.
```

### Other flags
- `--instance-type gpu_1x_h100_pcie` — override default `gpu_2x_a100`.
- `--git-branch some-branch` — override default `main2`.

## How it works (flow)

```
local                                        lambda cloud
─────                                        ────────────
lambda_launch.py
  ├─ POST /instance-operations/launch   ───▶ provisioning
  ├─ poll GET /instances/<id>           ◀── status=active, ip=...
  ├─ ssh-wait
  ├─ scp lambda_bootstrap.sh
  ├─ scp lambda_push_to_hf.py
  ├─ scp job.env (secrets + params)
  ├─ tmux new -s train 'bash bootstrap' ───▶  clone repo
  │                                          install deps
  │                                          run CPT.py or main.py
  │                                          lambda_push_to_hf.py --> HF Hub
  │                                          touch DONE_SUCCESS|DONE_FAILED
  ├─ poll (tail train.log + check sentinel)
  └─ POST /instance-operations/terminate ──▶ instance gone
```

## Failure modes & what to do

| Symptom | What to check |
|---|---|
| "no region has capacity for gpu_2x_a100" | Lambda is out of stock for that type. Retry later or use `--instance-type gpu_1x_a100_sxm4`. |
| SSH never comes up | `lambda_launch.py` waits 5 min. Check the Lambda dashboard — instance may be stuck provisioning. |
| Training hangs, you Ctrl+C locally | The remote job keeps running in tmux. The script proceeds to terminate anyway. Pass `--keep-alive` if you want to inspect. |
| `lambda_launch.py` crashes mid-run | **Check the Lambda dashboard and terminate the instance manually** — otherwise you keep paying. |
| HF push fails (token scope) | Ensure `HF_TOKEN` has write access to both target repos, and the repo namespace matches your account. |

## Cost guardrails
- Default flow always terminates the instance — even on `KeyboardInterrupt` and
  `DONE_FAILED`. The only path that *doesn't* terminate is `--keep-alive`.
- If the orchestrator itself dies (e.g. machine reboots), instances will keep
  running. Always double-check the Lambda dashboard after an unusual exit.
