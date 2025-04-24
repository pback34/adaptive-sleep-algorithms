

# Objective

The objective of this project is to collect as much sensor data as possible simultaneously. This includes all possible body location and signals that we are considering in the final product. We can prototype a unified device that contains sensors for all of the signal types we are considering, and in a form factor that can be used at any body location we are considering. We then capture data on ourselves and family members for as many nights as possible.

Additionally, while collecting data from our own sensors, we will need to simultaneously record data from a reference device to enable accurate labeling of the sleep stages for algorithm training and validation.

*Note: This document is still work in progress*


## Signals to Collect

See [[Task-P01-01-DecideSignals]]

The minimum set of signals we should acquire in our core sensor are:
- Accelerometer
- Gyroscope
- PPG

Optional, nice-to have sensor additions are:
- EDA (Electrodermal Activity)
- Temperature
- ECG
- EEG

## Derived Signals

Depending on the location of the sensor on the body, various derived signals can be acquired:

| Body Location   | Sensor        | Derived Signal                                     |
| --------------- | ------------- | -------------------------------------------------- |
| Chest + Abdomen | Accelerometer | Sleep Position, Respiratory Effort/Rate            |
| Chest           | PPG           | Heart Rate (possibly), Respiratory Rate (possibly) |
| Chest           | ECG           | Heart Rate, HRV, Respiratory Rate                  |
| Wrist           | Accelerometer | Arm Position, Arm Motion                           |
| Wrist           | PPG           | Heart Rate, SpO2 (possibly)                        |
| Wrist           | EDA           | EDA (standard location is forearm)                 |
| Foot            | Accelerometer | Body position, foot/leg movement                   |
| Foot            | PPG           | Heart Rate                                         |


## Recommended Body Locations

1. Wrist: This is standard and should be used as potential starting point
2. Chest + Abdomen: Using a single sensors, with dual accelometers (extension to the abdomen) to obtain respiratory effort and sleep position
3. Foot or ankle: less common location that may have some advantages over wrist (movement may)
	1. Potential issues: may be more difficult to measure heart rate
	2. Advantages: 
		- could determine body position more accurately
		- Can monitor restless leg syndrome (periodic limb movements during sleep)
4. EEG on Forehead: This provides the EEG data to use as reference data for sleep staging.

## Reference Signals

An EEG + Reference device should be used initially to determine which reference device (consumer wearable such as a ring or wrist wearable) can serve as the reference source for our training data. We can analyze the EEG ourselves to determine brainwave states and map to sleep stages (in-house PSG). The current options for EEG-based reference devices are the following:
- https://www.enchantedwave.com/: provides raw EEG data and analysis tools for sleep staging at affordable prices
- https://elemindtech.com/products/elemind: provides EEG-based sleep staging but likely limited raw data. The advantage of this one is that is provides the sleep staging which is more accurate than most wearables, so it could be quicker path to getting reference data
- https://www.pluxbiosignals.com/collections/biosignals-for-research/products/4-channel-biosignals-kit: Can use as a research-grade physiological monitoring system for both EEG and other signals

Any other devices we have available that provide sleep staging are also valuable to compare our results to, such as Fitbits, Garmin watches, and other wearables such as Oura.






