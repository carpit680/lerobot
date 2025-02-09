#!/usr/bin/env python3
import subprocess

def main():
    # Build the two commands as strings.

    # command0 = (
    #     'rm -rf /Users/arpit/.cache/huggingface/lerobot/carpit680/eval_giraffe_sock_demo_1_5 && '
    #     'python lerobot/scripts/control_robot.py '
    #     '--robot.type=so100 '
    #     '--control.type=record '
    #     '--control.fps=30 '
    #     '--control.single_task="Grasp a sock off the floor." '
    #     '--control.repo_id=carpit680/eval_giraffe_sock_demo_1_5 '
    #     '--control.tags=\'["giraffe","demo"]\' '
    #     '--control.warmup_time_s=1 '
    #     '--control.episode_time_s=20 '
    #     '--control.reset_time_s=1 '
    #     '--control.num_episodes=1 '
    #     '--control.push_to_hub=false '
    #     '--control.policy.path=/Users/arpit/Projects/lerobot/outputs/train/act_giraffe_task1/checkpoints/last/pretrained_model'
    # )
    command1 = (
        'rm -rf /Users/arpit/.cache/huggingface/lerobot/carpit680/eval_giraffe_sock_demo_1_5 && '
        'python lerobot/scripts/control_robot.py '
        '--robot.type=so100 '
        '--control.type=record '
        '--control.fps=30 '
        '--control.single_task="Grasp a sock off the floor." '
        '--control.repo_id=carpit680/eval_giraffe_sock_demo_1_5 '
        '--control.tags=\'["giraffe","demo"]\' '
        '--control.warmup_time_s=1 '
        '--control.episode_time_s=10 '
        '--control.reset_time_s=1 '
        '--control.num_episodes=1 '
        '--control.push_to_hub=false '
        '--control.policy.path=/Users/arpit/Projects/lerobot/outputs/train/act_giraffe_sock_demo_1/checkpoints/last/pretrained_model'
    )

    command2 = (
        'rm -rf /Users/arpit/.cache/huggingface/lerobot/carpit680/eval_giraffe_sock_demo_1_5 && '
        'python lerobot/scripts/control_robot.py '
        '--robot.type=so100 '
        '--control.type=record '
        '--control.fps=30 '
        '--control.single_task="Drop a sock into the bin." '
        '--control.repo_id=carpit680/eval_giraffe_sock_demo_1_5 '
        '--control.tags=\'["giraffe","demo"]\' '
        '--control.warmup_time_s=1 '
        '--control.episode_time_s=10 '
        '--control.reset_time_s=1 '
        '--control.num_episodes=1 '
        '--control.push_to_hub=false '
        '--control.policy.path=/Users/arpit/Projects/lerobot/outputs/train/act_giraffe_sock_demo_2/checkpoints/last/pretrained_model'
    )

    # Wait for the user to press Enter to run the first command.
    input("Press Enter to run the first command...")
    print("Executing first command:")
    print(command1)
    subprocess.run(command1, shell=True, check=True)
    print("First command finished.")

    # Wait for the user to press Enter to run the second command.
    input("Press Enter to run the second command...")
    print("Executing second command:")
    print(command2)
    subprocess.run(command2, shell=True, check=True)
    print("Second command finished.")

if __name__ == "__main__":
    main()
