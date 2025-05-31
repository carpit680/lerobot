import math
import serial
import collections
import numpy as np  # For median computation

class AS5600Sensor:
    def __init__(self, serial_port='/dev/cu.usbserial-120', baud_rate=115200):
        """
        Initialize the AS5600 sensor class.
        """
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.custom_zero = [2481, 12, 302, 4090, 3094, 2307]
        self.esp = serial.Serial(serial_port, baud_rate, timeout=1)
        self.dummy_angles = [0.0] * 6

        # Moving median filter deques (one for each channel)
        self.window_size = 10
        self.angle_windows = [collections.deque(maxlen=self.window_size) for _ in range(6)]

        print("AS5600 Sensor class has been Initialized")

    def convert_raw_to_degrees(self, raw_values, references):
        
        degrees = [0.0] * len(raw_values)
        for index, value in enumerate(raw_values):
            adjusted_value = (value - references[index] + 4096) % 4096
            degrees_value = (adjusted_value / 4096.0) * 360.0
            degrees[index] = (degrees_value + 180) % 360 - 180
        return degrees

    def map_value(self, x, in_min, in_max, out_min, out_max):
        return max(out_min, min(out_max, (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min))

    # def apply_median_filter(self, angle_values):
    #     """
    #     Update the median filter window and return the filtered values.
    #     """
    #     filtered_angles = []
    #     for i, angle in enumerate(angle_values):
    #         self.angle_windows[i].append(angle)
    #         median_filtered = float(np.median(self.angle_windows[i]))
    #         filtered_angles.append(median_filtered)
    #     return filtered_angles

    def apply_median_filter(self, angle_values):
        """
        Update the median filter window and return the filtered values.
        Uses angle unwrapping to avoid wrap-around artifacts.
        """
        filtered_angles = []

        for i, angle in enumerate(angle_values):
            # Append new angle
            self.angle_windows[i].append(angle)

            # Convert window to numpy array for easy manipulation
            angles_window = np.array(self.angle_windows[i])

            # Unwrap angles to prevent wrap-around artifacts
            angles_unwrapped = np.unwrap(np.deg2rad(angles_window))  # convert to radians and unwrap
            median_unwrapped = np.median(angles_unwrapped)

            # Convert back to degrees
            median_deg = math.degrees(median_unwrapped)

            # Wrap back to -180 to 180
            median_deg = (median_deg + 180) % 360 - 180

            filtered_angles.append(median_deg)

        return filtered_angles
    def clip_angle(self, angle, min_angle, max_angle):
        if angle >= max_angle:
            return max_angle
        elif angle <= min_angle:
            return  min_angle
        return angle
        
        
    def read_sensor_data(self):
        try:
            data = self.esp.readline().decode('utf-8').strip()
            if data:
                raw_values = list(map(int, data.split(",")))
                angles = self.convert_raw_to_degrees(raw_values, self.custom_zero)

                # Apply median filter
                median_filtered_angles = self.apply_median_filter(angles)
                
                gripper_value = self.map_value(abs(median_filtered_angles[5]), 0.0, 114.0, 0, 20)
                self.dummy_angles = [
                    self.clip_angle(median_filtered_angles[0], -90.0, 90.0),
                    abs(median_filtered_angles[1]),
                    abs(median_filtered_angles[2]),
                    self.clip_angle(median_filtered_angles[3], -90.0, 90.0), #rotate
                    -median_filtered_angles[4],
                    gripper_value
                ]
                # print([round(angle, 2) for angle in self.dummy_angles])
                # print(f"Raw Angles: {raw_values[1]}, Filtered Angles: {self.dummy_angles[1]}")
                return self.dummy_angles

        except serial.SerialException as e:
            print(f"Error from AS5600: {e}")
            return None
        except ValueError:
            print("Error from AS5600: Invalid data received.")
            return None
        except IndexError:
            print("Error from AS5600: Index out of range.")
            return None
        except Exception as e:
            print(f"Error from AS5600: {e}")
            return None

# Example usage
if __name__ == "__main__":
    sensor = AS5600Sensor(serial_port='/dev/cu.usbserial-120', baud_rate=115200)

    try:
        while True:
            angles = sensor.read_sensor_data()
            if angles:
                print([round(angle, 2) for angle in angles])
                pass
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
        sensor.esp.close()
