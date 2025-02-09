import time
import torch
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_utils import predict_action

def main():
    # ----- Define robot configuration -----
    # Adjust the parameters as needed for your robot.
    robot_config = {
        "robot_type": "so100",  # example robot type; change if needed
        # Add any other robot-specific configuration parameters here
    }
    # Instantiate the robot
    robot = make_robot_from_config(robot_config)
    
    # Connect to the robot (this will initialize sensors, cameras, etc.)
    print("Connecting to robot...")
    robot.connect()
    
    # ----- Define policy configuration -----
    # Provide the path to your pretrained ACT policy checkpoint.
    policy_config = {
        "path": "path/to/pretrained_model",  # update with your actual checkpoint directory
        "type": "act",  # indicates that this is an ACT policy
        # Include any additional policy parameters if required
    }
    
    # Set the device for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the pretrained policy using LeRobot’s factory method.
    print("Loading policy...")
    policy = make_policy(policy_config, device=device)
    policy.eval()  # ensure the policy is in evaluation mode
    
    # (Optional) Allow a short pause for sensors/cameras to stabilize.
    time.sleep(2)
    
    # ----- Execute the policy for 60 seconds -----
    execution_time = 60  # seconds
    fps = 30             # control loop frequency (adjust as needed)
    dt = 1.0 / fps
    print("Starting policy execution for 60 seconds...")
    
    start_time = time.time()
    while time.time() - start_time < execution_time:
        loop_start = time.time()
        
        # Capture the current observation from the robot.
        # (The robot is expected to have a method 'capture_observation' that returns a dict.)
        observation = robot.capture_observation()
        
        # Compute the next action.
        # The 'predict_action' utility converts image observations (if any) to the
        # required tensor format, adds a batch dimension, and then calls policy.select_action.
        action = predict_action(observation, policy, device, use_amp=False)
        
        # Send the action to the robot.
        robot.send_action(action)
        
        # Wait to maintain the desired control loop frequency.
        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)
    
    print("Policy execution complete. Disconnecting robot...")
    robot.disconnect()

if __name__ == "__main__":
    main()
