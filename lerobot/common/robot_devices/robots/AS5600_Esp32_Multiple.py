import serial

class AS5600Sensor:
    def __init__(self, serial_port='/dev/ttyUSB0', baud_rate=115200, custom_zero=None):
        """
        Initialize the AS5600 sensor class.

        Args:
            serial_port (str): Serial port for communication (default: '/dev/ttyUSB0').
            baud_rate (int): Baud rate for serial communication (default: 115200).
            custom_zero (list): List of reference values for each sensor (default: None).
        """
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.custom_zero = custom_zero if custom_zero else [0] * 6  # Default zero for each sensor
        self.esp32 = serial.Serial(serial_port, baud_rate, timeout=1)

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

    def read_sensor_data(self):
        """
        Continuously read raw sensor data from the ESP32 and return the angles in degrees.
        This function will run indefinitely and return the angles as a list.
        """
        angles = []
        try:
            while True:
                data = self.esp32.readline().decode('utf-8').strip()  # Read the data from the ESP32
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
        finally:
            self.esp32.close()

# Example usage
if __name__ == "__main__":
    # Initialize the AS5600 sensor class with reference values for each sensor
    custom_zero = [2280, 845, 3500, 590, 2050, 1330] # Example custom zero values for each sensor
    sensor = AS5600Sensor(serial_port='/dev/ttyUSB0', baud_rate=115200, custom_zero=custom_zero)
    
    # Continuously read sensor data and return a list of angles
    angles = sensor.read_sensor_data()
    print(angles)  # You can print the angles or process them as needed



#custom_zero = [2280, 845, 3500, 590, 2050, 1330]
