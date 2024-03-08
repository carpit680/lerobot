# LeRobot

## Installation

Create a virtual environment with python 3.10, e.g. using `conda`:
```
conda create -y -n lerobot python=3.10
conda activate lerobot
```

[Install `poetry`](https://python-poetry.org/docs/#installation) (if you don't have it already)
```
curl -sSL https://install.python-poetry.org | python -
```

Install dependencies
```
poetry install
```

If you encounter a disk space error, try to change your tmp dir to a location where you have enough disk space, e.g.
```
mkdir ~/tmp
export TMPDIR='~/tmp'
```


## Usage


### Train

```
python lerobot/scripts/train.py \
hydra.job.name=pusht \
env=pusht
```

### Visualize offline buffer

```
python lerobot/scripts/visualize_dataset.py \
hydra.run.dir=tmp/$(date +"%Y_%m_%d") \
env=pusht
```

### Visualize online buffer / Eval

```
python lerobot/scripts/eval.py \
hydra.run.dir=tmp/$(date +"%Y_%m_%d") \
env=pusht
```


## TODO

If you don't know how to contribute or want to know the next features we working on, look on this project page: [LeRobot TODO](https://github.com/users/Cadene/projects/1)

Ask [Remi Cadene](re.cadene@gmail.com) for access if needed.


## Profile

**Example**
```python
from torch.profiler import profile, record_function, ProfilerActivity

def trace_handler(prof):
    prof.export_chrome_trace(f"tmp/trace_schedule_{prof.step_num}.json")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=3,
    ),
    on_trace_ready=trace_handler
) as prof:
    with record_function("eval_policy"):
        for i in range(num_episodes):
            prof.step()
```

```bash
python lerobot/scripts/eval.py \
pretrained_model_path=/home/rcadene/code/fowm/logs/xarm_lift/all/default/2/models/final.pt \
eval_episodes=7
```

## Contribute

**Style**
```
# install if needed
pre-commit install
# apply style and linter checks before git commit
pre-commit run -a
```

**Tests**

Install [git lfs](https://git-lfs.com/) to retrieve test artifacts (if you don't have it already).
On Mac:
```
brew install git-lfs
git lfs install
```

On Ubuntu:
```
sudo apt-get install git-lfs
git lfs install
```

Pull artifacts if they're not in [tests/data](tests/data)
```
git lfs pull
```

When adding a new dataset, mock it with
```
python tests/scripts/mock_dataset.py --in-data-dir data/<dataset_id> --out-data-dir tests/data/<dataset_id>
```

Run tests
```
DATA_DIR="tests/data" pytest -sx tests
```
