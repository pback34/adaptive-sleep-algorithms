#decision 

# Task: Decide on which BLE-enabled MCU to use for data collection experiment

---
## Answer

### MCU
**Nordic nRF5340** (Dual-core Bluetooth 5.3 SoC)

- Ultra-low power ARM Cortex-M33 dual-core processor.
- BLE 5.3 for low-power wireless communication.
- Integrated power management features.
- On-chip secure execution environment for data protection.
### Power Management

**Texas Instruments BQ25155**

- Ultra-small, highly integrated power management IC (PMIC).
- Supports Li-ion/Li-Po rechargeable batteries.
- Low quiescent current for efficient battery management.
- Built-in thermal and voltage protection.

### Wireless Charging

**TI BQ51050B (Wireless Power Receiver IC)**

- Qi-compatible wireless charging.
- Compact size with integrated rectifier.
- Low power loss for efficient charging.

### Battery

**Rechargeable Li-Po Battery (100mAh - 300mAh, depending on form factor constraints)**

- Slim and lightweight.
- Overcharge and short circuit protection.

### Connectivity & Storage

- **Bluetooth 5.3** for low-power wireless communication.
- **On-board SPI Flash (8MB)** for data storage and offline logging.

---
# Supporting Research


For developers seeking full hardware control, several development kits stand out as suitable options for wearable sensor projects:

## STMicroelectronics STEVAL-MKSBOX1V1 (SensorTile.box)

This kit offers extensive customization and sensor capabilities:

- Multiple integrated sensors: accelerometer, gyroscope, magnetometer, pressure, temperature, humidity
- Bluetooth connectivity for wireless data transmission
- Expert Mode for custom app development
- Firmware programming and debugging interface
- STM32 Open Development Environment (ODE) compatibility
- AI function pack with neural network libraries[1][8]

## Shimmer3 Bridge Amplifier+ Development Kit

This versatile kit provides:

- Raw data capture from force, resistance, and kinematic sensors
- Two Shimmer3 Bridge Amplifier+ units
- 9-axis IMU (accelerometer, gyroscope, magnetometer)
- Altimeter
- Bluetooth and local storage options
- Compatibility with LabVIEW, MATLAB, C#, and Java/Android APIs[2][3]

## ESP32 Wearable Development Board (Whadda WMW101)

This board offers:

- ESP32 microcontroller with built-in Wi-Fi and Bluetooth
- Compact form factor suitable for wearable designs
- Full access to GPIO pins for sensor integration
- Arduino IDE compatibility for easy programming[6]

## Lattice iCE40 Ultra Wearable Development Platform

For advanced developers requiring FPGA capabilities:

- iCE40 Ultra FPGA and MachXO2 FPGA
- Integrated sensors
- 1.54" LCD display
- Compact watch form factor
- Ideal for low-power, always-on applications[7]

These development kits provide the necessary hardware access and flexibility for creating custom wearable sensor solutions. They offer varying levels of integration and complexity, allowing you to choose based on your specific requirements for sensor types, processing power, and form factor.

Citations:
[1] https://www.st.com/en/evaluation-tools/steval-mksbox1v1.html
[2] https://www.shimmersensing.com/product/consensys-bridge-amplifier-development-kit/
[3] https://www.shimmersensing.com/wearable-sensor-products/development-kits/
[4] https://www.movesense.com/product/movesense-developer-kit-2-0/
[5] https://www.adafruit.com/category/194
[6] https://whadda.com/product/esp32-wearable-development-board-wmw101/
[7] https://www.latticesemi.com/products/developmentboardsandkits/ice40ultrawearabledevelopmentplatform
[8] https://www.mouser.com/new/stmicroelectronics/stm-steval-mksbox1v1-sensortile-box-kit/

---
Answer from Perplexity: pplx.ai/share


For developers seeking full hardware control through development kits with raw sensor access, these platforms stand out as top contenders:

## Modular Sensor Development Kits
**1. Shimmer Consensys Bundle**  
- **Sensors**: 9DoF IMU, ECG, EMG, GSR, PPG, altimeter  
- **Key Features**:  
  - Open-source firmware ([GitHub](https://github.com/Shimmer-Sensing))  
  - Raw data streaming via Bluetooth/BLE (up to 512Hz)  
  - Mesh networking capabilities  
  - 3.5mm expansion port for custom analog/digital sensors  
  - Includes biophysical leads and electrodes[1][7]  

**2. Renesas DA14585 IoT Multi-Sensor Kit**  
- **Sensors**: 15-DOF (6-axis IMU, magnetometer, environmental, proximity/light)  
- **Control Features**:  
  - SmartFusion software library for low-level calibration  
  - Amazon Alexa voice integration  
  - BLE 5.0 with 300m range  
  - Arm Cortex-M0 programming access[5]  

**3. TE Connectivity Mikroe Click Boards**  
- **Modular System**:  
  - Pressure, temp, humidity sensors on mikroBUS™ standard  
  - Pre-calibrated digital sensors  
  - I²C interface with 5V level shifting  
  - Compatible with 100+ MCU platforms[8]  

## Open-Source Hardware Platforms
| Platform                 | Core Features                     | Sensor Expansion          |
| ------------------------ | --------------------------------- | ------------------------- |
| OpenHealth (TI CC2650)   | Medical-grade API control         | ECG + 3D accelerometer    |
| SparkFun OpenLog Artemis | ARM Cortex-M4F + BLE              | 9DoF IMU breakout support |
| OSC Wearable Project     | ESP8266-based, GitHub open design | ADXL345 accelerometer     |

## Key Technical Considerations
- **Firmware Access**: Shimmer and Renesas provide direct register-level control through their SDKs  
- **Power Management**: DA14585 claims world's lowest power consumption (0.3μA sleep)  
- **Expansion**: Shimmer's 3.5mm jack supports custom analog front-ends  
- **Certification**: TE Connectivity boards come pre-certified for FCC/CE  

For maximum flexibility, the **Shimmer Consensys Bundle** provides clinical-grade biophysical sensing with true open-source firmware. Those needing environmental sensing should consider **Renesas' 15-DOF kit**, while **TE Click Boards** offer plug-and-play sensor modularity. Open-source alternatives like the **SparkFun Artemis** platform enable complete hardware customization at the PCB level[6][8].

Citations:
[1] https://www.shimmersensing.com/product/consensys-bundle-development-kit/
[2] https://www.st.com/en/evaluation-tools/steval-mksbox1v1.html
[3] https://elab.ece.wisc.edu/wp-content/uploads/sites/1634/2019/02/OpenHealth_preprint.pdf
[4] https://openresearchsoftware.metajnl.com/articles/10.5334/jors.454
[5] https://www.renesas.com/en/products/wireless-connectivity/bluetooth-low-energy/da14585iotmsenskt-smartbond-da14585-bluetooth-low-energy-iot-multi-sensor-development-kit
[6] https://discuss.tinyml.seas.harvard.edu/t/looking-for-a-open-source-wearable-device/784
[7] https://www.shimmersensing.com/product/consensys-gsr-development-kits/
[8] https://www.te.com/en/products/sensors/sensor-development-boards-and-evaluation-kits.html
[9] https://mahilab.rice.edu/sites/default/files/publications/Snaptics_2021.pdf
[10] https://community.troikatronix.com/topic/3402/osc-wireless-wearable-sensors
[11] https://verisense.net/blog/raw-sensor-data

---
Answer from Perplexity: pplx.ai/share
