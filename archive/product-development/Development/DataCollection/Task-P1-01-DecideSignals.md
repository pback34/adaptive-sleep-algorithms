#decision
# Task: Decide on which signals to measure


## Proposals

### Proposal for Phase 1 data collection
*by Peter Backeris*

In phase 1, we should collect more data than we anticipate that we will need in final product. The purpose is to have enough data to make a decision about which combinations of signals provides sufficient accuracy, speed, and resolution into sleep staging. There are 2 core signals that will be likely to be needed as a minimum: **accelerometer** and **PPG** (or other signal that tracks the heart beat such as ECG). Additional sensors can provide further improvements to accuracy. The next one that I would recommend to add value to this would be **respiratory rate**, followed by **estimated core temperature**.

We should incorporate a 6 to 9-DOF accelometer/gyro/magnetometer sensor, a skin-facing temperature sensor, and an ambient facing sensor as a minimum for the core board design. A PPG module (from Maxim) should be integrated into the board as well, or as a flex-cable based extension. An accelerometer on a flex cable extension should also be added. 

Additionally, we should have an ambient light sensor (this is already monitored by the Hatch product, but it may be convenient just to have for those that don't have a Hatch product, and for making data collection easier).



---
## Consider Adding EDA (Electrodermal Activity) to the collection?

### **Components of an EDA Sensor**
An electrodermal activity (EDA) sensor typically consists of the following components:

1. **Electrodes**:
   - **Material**: Commonly made of **Ag/AgCl (silver/silver-chloride)** for their stability, low noise, and ability to minimize polarization artifacts.
   - **Placement**: Often placed on the palms, fingers, or soles of the feet, where sweat glands are densest.
   - **Types**: 
     - Disposable electrodes with conductive gel for better signal fidelity.
     - Reusable electrodes embedded in straps or patches.

2. **Amplifier**:
   - Boosts the weak EDA signal for further processing.
   - Features low noise and high input impedance to maintain signal quality.

3. **Analog-to-Digital Converter (ADC)**:
   - Converts the amplified analog EDA signal into a digital format for processing.
   - High-resolution ADCs (e.g., 12-bit or 14-bit) are preferred for capturing subtle changes in skin conductance.

4. **Signal Conditioning Circuitry**:
   - Includes filters to remove noise and isolate relevant frequency bands:
     - **Low-pass filters**: Typically with a cutoff frequency of around 1.5 Hz to capture phasic responses.
     - **High-pass filters**: To eliminate slow drifts in the tonic component.

5. **Current Source**:
   - Supplies a small, constant DC or AC current to measure skin conductance without damaging sweat glands or introducing nonlinearities.

6. **Microcontroller/Processor**:
   - Processes raw EDA signals, separates tonic and phasic components, and transmits data for analysis.
   - May include algorithms for peak detection and feature extraction.

7. **Wireless Data Transmission Module (optional)**:
   - Enables real-time data transfer via Bluetooth or other wireless protocols in wearable devices.

---

### **Off-the-Shelf Sensors and ICs for EDA**
There are several off-the-shelf solutions available for building or integrating EDA sensors:

#### **1. Integrated Circuits (ICs)**
- **MAX30001G (Analog Devices)**:
  - Features a biopotential channel and a bioimpedance channel optimized for galvanic skin response (GSR) and electrodermal activity.
  - Includes built-in filtering, calibration, and low-power operation suitable for wearables.
  - Supports both 2-electrode and 4-electrode configurations.

#### **2. Ready-to-Use Sensors**
- **Biosignalsplux EDA Sensor**:
  - Measures skin resistance with pre-conditioned analog output.
  - Medical-grade raw data output with high signal-to-noise ratio.
  - Compatible with specific acquisition systems like BITalino.

- **Movisens EdaMove 4**:
  - Combines EDA measurement with motion tracking and skin temperature sensing.
  - Offers a wide measurement range (2 µS–100 µS) with DC exosomatic methods.

#### **3. Modular Kits**
- **BIOPAC Systems**:
  - Provides modular systems like the MP160 with dedicated amplifiers (e.g., EDA100C/EDA100D) for high-resolution EDA data collection.
  - Includes software tools for signal analysis and integration with other physiological measures.

- **ADInstruments Electrodermal Activity Kit**:
  - Includes finger electrodes, amplifiers, and thermistor pods for comprehensive physiological monitoring.

---

### **Applications of Off-the-Shelf Solutions**
- Prototyping biomedical devices.
- Stress monitoring in wearables.
- Research in psychophysiology, human-computer interaction, and affective computing.

By combining these components or using ready-made solutions, you can develop robust EDA systems tailored to specific applications.

Citations:
[1] https://pmc.ncbi.nlm.nih.gov/articles/PMC5677183/
[2] https://www.analog.com/en/products/max30001g.html
[3] https://www.robotshop.com/products/biosignalsplux-electrodermal-activity-eda-sensor
[4] https://www.analog.com/en/resources/technical-articles/design-development-and-evaluation-of-a-system-to-obtain-electrodermal-activity.html
[5] https://sensorwiki.org/sensors/galvanic_skin_response
[6] https://www.movisens.com/en/products/eda-and-activity-sensor/
[7] https://imotions.com/blog/learning/research-fundamentals/eda-peak-detection/
[8] https://www.pluxbiosignals.com/products/electrodermal-activity-eda-sensor-1
[9] https://synapse.patsnap.com/article/what-are-eda-stimulants-and-how-do-they-work
[10] https://www.pluxbiosignals.com/products/electrodermal-activity-eda-sensor
[11] https://www.biopac.com/wp-content/uploads/EDA-SCR-Analysis.pdf
[12] https://www.adinstruments.com/products/electrodermal-activity-kit
[13] https://www.biopac.com/wp-content/uploads/EDA-Guide.pdf
[14] https://imotions.com/blog/learning/research-fundamentals/eda/
[15] https://infoscience.epfl.ch/server/api/core/bitstreams/513312f3-c96d-4fb0-a27e-bc947f100bf1/content
[16] https://www.keysight.com/blogs/en/tech/educ/eda-tools
[17] https://www.prnewswire.com/news-releases/electronic-design-automation-eda-market-to-grow-by-usd-8-7-billion-2024-2028-with-ais-rising-impact-on-trends---technavio-report-302250801.html
[18] https://www.keysight.com/blogs/en/tech/educ/2024/electronic-design-automation
[19] https://us.metoree.com/categories/4165/

---
Answer from Perplexity: pplx.ai/share








