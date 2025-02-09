#!/usr/bin/env python3
import subprocess

def main():
    # Build the command as a single string. Note that shell expansion (e.g. ${HF_USER})
    # requires shell=True and that the HF_USER environment variable is set.
    command = (
        'rm -rf /Users/arpit/.cache/huggingface/lerobot/carpit680/eval_giraffe_sock_demo_1_5 && '
        'python lerobot/scripts/control_robot.py '
        '--robot.type=so100 '
        '--control.type=record '
        '--control.fps=30 '
        '--control.single_task="Grasp a sock off the floor." '
        '--control.repo_id=carpit680/eval_giraffe_sock_demo_1_5 '
        '--control.tags=\'["giraffe","demo"]\' '
        '--control.warmup_time_s=1 '
        '--control.episode_time_s=20 '
        '--control.reset_time_s=1 '
        '--control.num_episodes=1 '
        '--control.push_to_hub=false '
        '--control.policy.path=/Users/arpit/Projects/lerobot/outputs/train/act_giraffe_task1/checkpoints/last/pretrained_model'
    )
    
    print("Executing command:")
    print(command)
    
    # Run the command in the shell.
    subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    main()
