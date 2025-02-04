"""import serial
import time

class AS5600Sensor:
    def __init__(self, serial_port='/dev/ttyUSB0', baud_rate=115200, custom_zero=None):

        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.custom_zero = custom_zero if custom_zero else [0] * 6  # Default zero for each sensor
        self.esp32 = serial.Serial(serial_port, baud_rate, timeout=1)
        print("AS5600 Sensor class has been Initialized")

    def convert_raw_to_degrees(self, raw_value, reference):

        adjusted_value = (raw_value - reference + 4096) % 4096
        degrees = (adjusted_value / 4096.0) * 360.0  # Convert to degrees (0-360)
        degrees = (degrees + 180) % 360 - 180  # Normalize to -180 to 180
        return degrees

    def read_sensor_data(self):

        angles = []
        try:
           # while True:
            data = self.esp32.readline().decode('utf-8').strip()  # Read the data from the ESP32
            print("Lodu")
            if data:  # If there's data, process it
                raw_values = list(map(int, data.split(",")))  # Split and convert to integers

                # Convert each raw value to degrees using the reference values
                angles = [self.convert_raw_to_degrees(raw_values[i], self.custom_zero[i]) for i in range(len(raw_values))]
                dummy_angles = [angles[0],0.0,0.0,0.0,0.0,0.0]
                
                return dummy_angles  # Return the list of angles

        except serial.SerialException as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("Exiting...")
       # finally:
        #    self.esp32.close()

# Example usage
if __name__ == "__main__":
    # Initialize the AS5600 sensor class with reference values for each sensor
    custom_zero = [50, 845, 3500, 590, 2050, 1330] # Example custom zero values for each sensor
    sensor = AS5600Sensor(serial_port='/dev/ttyUSB0', baud_rate=115200, custom_zero=custom_zero)
    
    # Continuously read sensor data and return a list of angles
    while True:

        angles = sensor.read_sensor_data()
        print(angles)  # You can print the angles or process them as needed
        time.sleep(0.5)

"""

#custom_zero = [2280, 845, 3500, 590, 2050, 1330]

"""
import serial
import time

class AS5600Sensor:
    def __init__(self, serial_port='/dev/ttyUSB0', baud_rate=115200):

        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.custom_zero = [50, 845, 3500, 590, 2050, 1330]  # Moved inside the class
        self.esp32 = serial.Serial(serial_port, baud_rate, timeout=1)
        print("AS5600 Sensor class has been Initialized")

    def convert_raw_to_degrees(self, raw_value, reference):

        adjusted_value = (raw_value - reference + 4096) % 4096
        degrees = (adjusted_value / 4096.0) * 360.0  # Convert to degrees (0-360)
        degrees = (degrees + 180) % 360 - 180  # Normalize to -180 to 180
        return degrees

    def read_sensor_data(self):

        angles = []
        try:
            # while True:
            data = self.esp32.readline().decode('utf-8').strip()  # Read the data from the ESP32
            print("Lodu")
            if data:  # If there's data, process it
                raw_values = list(map(int, data.split(",")))  # Split and convert to integers

                # Convert each raw value to degrees using the reference values
                angles = [self.convert_raw_to_degrees(raw_values[i], self.custom_zero[i]) for i in range(len(raw_values))]
                dummy_angles = [angles[0], 0.0, 0.0, 0.0, 0.0, 0.0]

                return dummy_angles  # Return the list of angles

        except serial.SerialException as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("Exiting...")
        # finally:
        #    self.esp32.close()

# Example usage
if __name__ == "__main__":
    # Initialize the AS5600 sensor class
    sensor = AS5600Sensor(serial_port='/dev/ttyUSB0', baud_rate=115200)

    # Continuously read sensor data and return a list of angles
    while True:
        angles = sensor.read_sensor_data()
        print(angles)  # You can print the angles or process them as needed
       # time.sleep(0.5)"""



import serial
import time

class AS5600Sensor:
    def __init__(self, serial_port='/dev/ttyUSB0', baud_rate=115200):
        """
        Initialize the AS5600 sensor class.

        Args:
            serial_port (str):             goal_pos = self.follower_arms[name].read("Present_Position")
            if tong_goal_pos is not None:
                # goal_pos[0] = tong_goal_pos[0] # works
                goal_pos[1] = tong_goal_pos[1]
                # goal_pos[2] = tong_goal_pos[2] # works
                # goal_pos[3] = tong_goal_pos[3] # works
                # goal_pos[4] = tong_goal_pos[4] # works
                # goal_pos[5] = tong_goal_pos[5] # worksSerial port for communication (default: '/dev/ttyUSB0').
            baud_rate (int): Baud rate for serial communication (default: 115200).
        """
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.custom_zero = [2280, 845, 3450, 590, 2990, 1330]  # Moved inside the class
        self.esp32 = serial.Serial(serial_port, baud_rate, timeout=1)
        self.dummy_angles=[0.0]*6
        print("AS5600 Sensor class has been Initialized")

    def convert_raw_to_degrees(self, raw_value, reference):
        """
        Convert raw AS5600 value (0-4095) to degrees with custom zero.

        Args:
            raw_value (int): Raw value from the AS5600 sensor.
            reference (int): Custom zero reference value.

        Returns:
            float: Angle in degrees, normalized to -180° to 180°.
        """
        adjusted_value = (raw_value - reference + 4096) % 4096
        degrees = (adjusted_value / 4096.0) * 360.0  # Convert to degrees (0-360)
        degrees = (degrees + 180) % 360 - 180  # Normalize to -180 to 180
        return degrees
    

    def map_value(self, x, in_min, in_max, out_min, out_max):
        """
        Linearly map a value from one range to another.

        Args:
            x (float): Input value.
            in_min (float): Minimum value of input range.
            in_max (float): Maximum value of input range.
            out_min (float): Minimum value of output range.
            out_max (float): Maximum value of output range.

        Returns:
            float: Mapped value.
        """
        return max(out_min, min(out_max, (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min))


    def read_sensor_data(self):
        """
        Read raw sensor data from the ESP32 and return the angles in degrees.
        """
        try:
            data = self.esp32.readline().decode('utf-8').strip()  # Read the data from the ESP32
            #print("Lodu")
            if data:  # If there's data, process it
                raw_values = list(map(int, data.split(",")))  # Split and convert to integers

                # Convert each raw value to degrees using the reference values
                angles = [self.convert_raw_to_degrees(raw_values[i], self.custom_zero[i]) for i in range(len(raw_values))]
                Gripper_value = self.map_value(raw_values[5], 1325, 2808, 0, 25)
                self.dummy_angles = [0.0,angles[1],-angles[2],-angles[3],-angles[4], Gripper_value ] #ang[2],[3] are negative

                return self.dummy_angles  # Return the list of angles

        except serial.SerialException as e:
            print(f"Error: {e}")
            return self.dummy_angles
        except ValueError:
            print("Invalid data received.")
            return self.dummy_angles

# Example usage
if __name__ == "__main__":
    # Initialize the AS5600 sensor class
    sensor = AS5600Sensor(serial_port='/dev/ttyUSB0', baud_rate=115200)

    try:
        while True:
            angles = sensor.read_sensor_data()
            if angles:
                print(angles)  # You can print the angles or process them as needed
            #time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
        sensor.esp32.close()  # Close serial port before exiting

