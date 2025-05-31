#!/usr/bin/env python3
import subprocess
from tts import KokoroTTS

tts_engine = KokoroTTS(lang_code='a', sample_rate=22050)

def main():
    # Build the two commands as strings.

    command1 = (
        'rm -rf /Users/arpit/.cache/huggingface/lerobot/carpit680/eval_giraffe_sock_demo_1_5 && '
        'python3 lerobot/scripts/control_robot.py '
        '--robot.type=so100 '
        '--control.type=record '
        '--control.fps=30 '
        '--control.single_task="Grasp a sock off the floor." '
        '--control.repo_id=carpit680/eval_giraffe_sock_demo_1_5 '
        '--control.tags=\'["giraffe","demo"]\' '
        '--control.warmup_time_s=1 '
        '--control.episode_time_s=40 '
        '--control.reset_time_s=1 '
        '--control.num_episodes=1 '
        '--control.push_to_hub=false '
        '--control.policy.path=/Users/arpit/Projects/lerobot/outputs/train/act_giraffe_sock_demo_1/checkpoints/last/pretrained_model'
    )

    command2 = (
        'rm -rf /Users/arpit/.cache/huggingface/lerobot/carpit680/eval_giraffe_sock_demo_1_5 && '
        'python3 lerobot/scripts/control_robot.py '
        '--robot.type=so100 '
        '--control.type=record '
        '--control.fps=30 '
        '--control.single_task="Drop a sock into the bin." '
        '--control.repo_id=carpit680/eval_giraffe_sock_demo_1_5 '
        '--control.tags=\'["giraffe","demo"]\' '
        '--control.warmup_time_s=1 '
        '--control.episode_time_s=40 '
        '--control.reset_time_s=1 '
        '--control.num_episodes=1 '
        '--control.push_to_hub=false '
        '--control.policy.path=/Users/arpit/Projects/lerobot/outputs/train/act_giraffe_sock_demo_2/checkpoints/last/pretrained_model'
    )
    input("Press Enter to proceed...")
    say="I see a kinda messy hall with a bunch of tools lying around and a person holding a phone."
    tts_engine.speak(say, voice='af_heart', speed=1)

    input("Press Enter to proceed...")
    say="Sure, where would you like me to start?"
    tts_engine.speak(say, voice='af_heart', speed=1)

    input("Press Enter to proceed...")
    say="Sure, but a single sock at the center is a little weird."
    tts_engine.speak(say, voice='af_heart', speed=1)

    input("Press Enter to run the first command...")
    subprocess.run(command1, shell=True, check=True)
    print("First command finished.")

    input("Press Enter to run the second command...")
    subprocess.run(command2, shell=True, check=True)
    print("Second command finished.")

    input("Press Enter to proceed...")
    say="Would that be all?."
    tts_engine.speak(say, voice='af_heart', speed=1)

if __name__ == "__main__":
    main()
