#!/usr/bin/env python3
import subprocess
from tts import KokoroTTS

tts_engine = KokoroTTS(lang_code='a', sample_rate=22050)

def main():

    command1 = (
        'rm -rf /Users/arpit/.cache/huggingface/lerobot/carpit680/giraffe_clean_desk && '
        'python3 lerobot/scripts/control_robot.py '
        '--robot.type=so100 '
        '--control.type=record '
        '--control.fps=30 '
        '--control.single_task="Clean up the desk." '
        '--control.repo_id=carpit680/giraffe_clean_desk '
        '--control.tags=\'["giraffe", "clean", "desk]\' '
        '--control.warmup_time_s=1 '
        '--control.episode_time_s=40 '
        '--control.reset_time_s=1 '
        '--control.num_episodes=1 '
        '--control.push_to_hub=false '
        '--control.policy.path=/Users/arpit/Projects/lerobot/outputs/train/giraffe_clean_desk/checkpoints/last/pretrained_model'
    )

    input("Press Enter to proceed...")
    say=""
    tts_engine.speak(say, voice='af_heart', speed=1)

    input("Press Enter to run the first command...")
    subprocess.run(command1, shell=True, check=True)
    print("Run finished.")

if __name__ == "__main__":
    main()
