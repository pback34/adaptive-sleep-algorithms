

# Question: Can we access raw/processed data in real time on existing wearable platforms to develop our own sleep app?


Access to raw sensor data in consumer wearables like the Apple Watch or competing devices is limited, but here’s a breakdown of current capabilities and constraints:  

### **Apple Watch**  
- **Accelerometer**:  
  Raw accelerometer data is accessible via Core Motion’s `CMMotionManager` for real-time sampling in apps[3][7][19]. Developers can retrieve XYZ-axis motion data at specified intervals, enabling basic movement analysis[3][7].  
  - Example code snippet:  
    ```swift
    let motionManager = CMMotionManager()
    motionManager.accelerometerUpdateInterval = 0.1
    motionManager.startAccelerometerUpdates(to: .main) { (data, error) in
        // Process acceleration data
    }
    ```
  - **Limitation**: Background access is restricted unless using SensorKit (for approved research studies)[15].  

- **PPG (Heart Rate Sensor)**:  
  No direct access to raw PPG waveforms. Heart rate data is available via HealthKit as preprocessed BPM values[1][12][15]. Third-party apps must rely on Apple’s aggregated metrics (e.g., heart rate variability, resting rate)[13][21].  

### **Competitors (Samsung, Garmin, Fitbit, WearOS)**  
- **Samsung Galaxy Watch**:  
  - PPG raw data requires a partnership with Samsung and access to their **Privileged Health SDK**[9][29].  
  - Accelerometer data is accessible via standard Android/WearOS APIs, but advanced sensors (e.g., SpO2) are often restricted[14][29].  

- **Garmin**:  
  - Limited raw PPG access. Some devices (e.g., Vivoactive 4) export interbeat intervals[5], but raw photoplethysmography signals are typically locked[10][35].  
  - Accelerometer data is accessible for custom apps via Connect IQ, but sampling rates vary by device[30][38].  

- **Fitbit**:  
  - No raw PPG/accelerometer access. Processed metrics (e.g., heart rate, step count) are available via Fitbit’s API or third-party platforms like Fitabase[6][11][28].  

- **WearOS (General)**:  
  - Google’s Health Services API provides aggregated motion/health metrics but not raw sensor streams[14][32].  
  - Exceptions exist for OEM-specific sensors (e.g., Samsung’s PPG), though documentation is sparse[9][14].  

### **Third-Party Sleep Apps**  
Most sleep-tracking apps (e.g., SleepWatch, Pillow) rely on:  
1. **Processed HealthKit/Garmin/Fitbit data** (e.g., heart rate, movement)[8][13].  
2. **Indirect metrics** like HRV or respiratory rate to infer sleep stages[21][24].  
3. **Manual user input** for sleep/wake times[8].  

### **Key Limitations**  
- **Raw PPG waveforms** are rarely exposed due to proprietary algorithms and hardware restrictions[1][12][20].  
- **Battery/performance concerns** limit real-time sensor access on wearables[6][14].  
- **OEM partnerships** are often required for advanced data (e.g., Samsung Privileged SDK)[9][29].  

For developers, workarounds include focusing on **processed intraday data** (e.g., 1-second heart rate samples via Fitbit[6] or Apple Health[15]) or exploring research-focused tools like SensorKit (Apple) or Fitabase (Fitbit)[6][15]. However, truly raw sensor access remains uncommon in consumer devices.

Citations:
[1] https://stackoverflow.com/questions/68174731/if-i-can-develop-app-for-apple-smartwatch-to-get-raw-ppg-data-and-implement-my-o
[2] https://kb.exist.io/article/46-how-do-i-see-my-raw-data-in-apple-health
[3] https://stackoverflow.com/questions/46756274/how-to-work-with-accelerometer-data-from-the-apple-watch
[4] https://stackoverflow.com/questions/28513191/is-it-possible-to-get-ppgphotoplethysmogram-data-from-a-smart-watch-via-androi
[5] https://www.reddit.com/r/Garmin/comments/gqomsp/extracting_raw_ppg_time_series_from_vivoactive_4/
[6] https://enterprise.fitbit.com/researchers/faqs/
[7] https://stackoverflow.com/questions/27002910/is-there-any-way-to-access-the-accelerometer-from-the-apple-watch
[8] https://help.cardiogram.com/article/138-how-can-i-track-sleep-time-on-an-apple-watch
[9] https://forum.developer.samsung.com/t/how-to-access-ppg-raw-values-on-galaxy-watch-4/23393
[10] https://forums.garmin.com/developer/connect-iq/f/discussion/1023/raw-sensor-data
[11] https://www.reddit.com/r/fitbit/comments/3o192v/happy_and_sad_getting_access_to_raw_fitbit_data/
[12] https://www.reddit.com/r/AppleWatch/comments/13slu7m/are_there_any_apps_that_use_the_ppg_heart_rate/
[13] https://support.apple.com/guide/watch/track-your-sleep-apd830528336/11.0/watchos/11.0
[14] https://www.reddit.com/r/WearOSDeveloper/comments/1csfomu/get_ppg_raw_data_from_wearos_smartwatch/
[15] https://forums.developer.apple.com/forums/thread/733186
[16] https://developer.apple.com/design/human-interface-guidelines/gyro-and-accelerometer
[17] https://developer.apple.com/documentation/sensorkit/srphotoplethysmogramsample
[18] https://pmc.ncbi.nlm.nih.gov/articles/PMC6930135/
[19] https://developer.apple.com/documentation/coremotion/getting-raw-accelerometer-events
[20] https://discussions.apple.com/thread/7102234
[21] https://support.apple.com/en-us/108906
[22] https://developer.apple.com/documentation/sensorkit/srphotoplethysmogramopticalsample
[23] https://pmc.ncbi.nlm.nih.gov/articles/PMC10980217/
[24] https://help.welltory.com/en/articles/9888090-new-sleep-analysis-for-apple-watch-users
[25] https://developer.apple.com/documentation/coremotion/
[26] https://apps.apple.com/us/app/sensor-logger/id1531582925
[27] https://www.reddit.com/r/AthlyticAppOfficial/comments/10st3dl/sleep_data_missing_or_seems_inaccurate/
[28] https://ilumivu.com/2017/04/researchers-guide-to-wearables-activity/
[29] https://developer.samsung.com/sdp/blog/en/2022/05/25/check-which-sensor-you-can-use-in-galaxy-watch-running-wear-os-powered-by-samsung
[30] https://forums.garmin.com/apps-software/mobile-apps-web/f/garmin-connect-web/152719/extracting-accelerometer-data
[31] https://community.fitbit.com/t5/SDK-Development/How-Do-I-Get-The-Raw-Data-From-My-FitBit-Watch/td-p/4485257
[32] https://developers.google.com/fit/android/sensors
[33] https://forum.quantifiedself.com/t/oldie-but-goldie-getting-full-access-to-fitbit-data-is-it-possible/2836
[34] https://pmc.ncbi.nlm.nih.gov/articles/PMC5887043/
[35] https://forums.garmin.com/developer/connect-iq/f/discussion/372713/raw-data-collection-from-garmin
[36] https://mhealth.jmir.org/2021/3/e25289
[37] https://github.com/GeoTecINIT/WearOSSensors
[38] https://developer.garmin.com/connect-iq/core-topics/sensors/
[39] https://forums.garmin.com/developer/connect-iq/f/discussion/256905/documentation-for-data-available-on-watches

---
Answer from Perplexity: pplx.ai/share

# Apple Watch
Here's a comprehensive breakdown of sensor data access through Apple's platforms:

## Accelerometer Data
- Raw accelerometer data is accessible through CoreMotion APIs at up to 100Hz in real-time when actively running a watch app[9]
- New Core Motion APIs provide access to 800Hz accelerometer data[10]
- The app must be actively running to access accelerometer data - there is no historical or background access without explicit user interaction[6]

## Heart Rate Data
- Real-time heart rate data is not directly accessible through raw PPG sensors[2]
- Heart rate data is only available through HealthKit after being processed and stored[25]
- There are inherent delays in heart rate data availability:
  - Data must first be gathered from sensors, processed, and saved to HealthKit
  - On iPhone, additional delay occurs during HealthKit store synchronization[25]

## Key Limitations
- Cannot access raw PPG sensor data directly[8]
- Cannot force continuous sensor monitoring or override sampling protocols[8]
- Background execution is limited - user must explicitly interact with the app to start sensor data collection[6]
- Heart rate data transmission can take up to 30 minutes to sync in some cases[8]

For real-time monitoring applications, the recommended approach is to:
4. Create a Watch app to collect sensor data
5. Use WatchConnectivity framework to transmit data to the iPhone app
6. Implement HKWorkoutSession for extended background execution[1]

Citations:
[1] https://stackoverflow.com/questions/73558381/how-do-you-configure-accelerometer-readings-in-real-time-applewatch
[2] https://forums.developer.apple.com/forums/thread/6549
[3] https://stackoverflow.com/questions/53780808/how-to-get-heart-rate-data-near-by-real-time-from-health-kit-in-ios
[4] https://www.researchandcare.org/faq/
[5] https://developer.apple.com/videos/play/wwdc2019/218/
[6] https://stackoverflow.com/questions/50765578/can-iphone-retrieve-accelerometer-data-from-the-watch-without-a-watch-app
[7] https://forums.developer.apple.com/forums/thread/733186
[8] https://developer.apple.com/forums/thread/658101
[9] https://developer.apple.com/videos/play/wwdc2023/10179/
[10] https://developer.apple.com/documentation/updates/healthkit
[11] https://www.researchgate.net/post/What_options_are_available_for_real_time_physical_activity_tracking_with_wearable_Accelerometers
[12] https://sahha.ai/platform/features/healthkit-data-sdk
[13] https://github.com/dwyl/learn-apple-watch-development/issues/35
[14] https://apple.stackexchange.com/questions/220016/possibility-of-getting-healthkit-data-from-third-party-sensors-in-real-time
[15] https://developer.apple.com/documentation/coremotion/getting-raw-accelerometer-events
[16] https://apps.apple.com/us/app/sensor-logger/id1531582925
[17] https://www.tryvital.com/wearables-api/apple-health-kit
[18] https://thryve.health/apple-health-kit-integration/
[19] https://developer.apple.com/videos/play/wwdc2019/218/
[20] https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/1615138-heartrate
[21] https://forums.developer.apple.com/forums/thread/736921
[22] https://www.reddit.com/r/iOSProgramming/comments/snf2eq/core_motion_sensor_data_is_it_difficult_to_access/
[23] https://www.youtube.com/watch?v=uzJXV_9IBoc
[24] https://forums.developer.apple.com/forums/thread/102292
[25] https://forums.developer.apple.com/forums/thread/756354
[26] https://support.apple.com/en-us/120277
[27] https://developer.apple.com/forums/thread/91788

---
Answer from Perplexity: https://www.perplexity.ai/search/do-any-wearable-watches-like-t-CpVrlgxMSQ6YWKh5kQHhCA?utm_source=copy_output


# Samung Wear OS Watch

For Samsung Wear OS watches, here's how sensor data access works:

## Raw Accelerometer Access
- Raw accelerometer data is accessible through standard Android Sensor APIs using Android Studio[1]
- Developers can capture real-time accelerometer data at high sampling rates
- The data can be recorded and transferred to a paired phone[15]

## Heart Rate Data
- Basic heart rate data is available through Wear Health Services API[7]
- Real-time heart rate monitoring is possible through the Health Services API, but with some limitations[11]
- Raw PPG (photoplethysmography) data requires Samsung Privileged Health SDK partnership[2][3]

## Samsung Privileged Health SDK
- Provides access to processed sensor data and advanced health metrics[2]
- Requires partnership approval through Samsung's program[14]
- Raw sensor access is restricted to protect proprietary algorithms[3]

## Key Limitations
- Standard apps cannot access raw PPG sensor data
- Real-time data transfer to phone requires custom implementation using Wearable Data Layer API[11]
- Background execution and continuous monitoring have restrictions
- Samsung Health SDK only provides processed data, not raw sensor output[3]

For developers wanting to work with sensor data, the recommended approach is to use Android's Health Services API for basic metrics, or apply for Samsung's Privileged Health partnership for advanced sensor access[14].

Citations:
[1] https://forum.developer.samsung.com/t/extracting-live-sensor-gyro-and-accelerometer-data-from-galaxy-watch4-in-real-time/18735
[2] https://developer.samsung.com/health/blog/en/transfer-heart-rate-from-galaxy-watch-to-a-phone
[3] https://forum.developer.samsung.com/t/samsung-health-sdk/21553
[4] https://www.dcrainmaker.com/2024/07/samsung-galaxy-watch-ultra-finally-accurate.html
[5] https://forum.developer.samsung.com/t/samsung-health-privileged-sdk/17663
[6] https://stackoverflow.com/questions/60613515/how-to-fetch-active-time-data-from-samsung-health-in-my-android-app
[7] https://developer.android.com/health-and-fitness/guides/basic-fitness-app/integrate-wear-os
[8] https://www.samsung.com/us/support/answer/ANS10003311/
[9] https://www.reddit.com/r/WearOS/comments/1e34n2y/wear_os_is_very_limited/
[10] https://forum.developer.samsung.com/t/unclear-partnership-program/17277
[11] https://stackoverflow.com/questions/75914712/how-to-send-real-time-sensor-data-from-wear-os-to-mobile
[12] https://forum.developer.samsung.com/t/heart-rate-complication-doesnt-display-data/27896
[13] https://www.androidpolice.com/google-wear-os-problems-solutions/
[14] https://hitconsultant.net/2024/01/08/samsung-expands-its-privileged-health-sdk-partner-program/
[15] https://forum.developer.samsung.com/t/samsung-watch-4-accelerometer-recorder/21473
[16] https://forum.developer.samsung.com/t/how-to-access-the-galaxy-watch-health-data-by-independent-application/20001
[17] https://forum.developer.samsung.com/t/get-ppg-via-health-platform-api/29712
[18] https://forum.developer.samsung.com/t/how-to-get-gyroscope-data-of-galaxy-watch-4-using-android-studio/22400
[19] https://www.samsung.com/latin_en/support/mobile-devices/measure-your-ecg-with-the-galaxy-watch-series/
[20] https://forum.developer.samsung.com/t/how-to-download-the-samsung-privileged-health-sdk-application-policy/30628
[21] https://www.reddit.com/r/WearOS/comments/12asfjo/probably_doesnt_exist_but_is_there_some_kind_of/
[22] https://stackoverflow.com/questions/70427283/how-to-read-real-time-ecg-data-in-samsung-galaxy-watch-4-wear-os
[23] https://developer.samsung.com/health/android/overview.html
[24] https://developer.samsung.com/sdp/blog/en/2022/05/25/check-which-sensor-you-can-use-in-galaxy-watch-running-wear-os-powered-by-samsung
[25] https://www.reddit.com/r/WearOS/comments/nuof5e/monitor_my_live_heart_rate_and_cast_to_android/
[26] https://support.google.com/wearos/answer/6056862?hl=en
[27] https://forum.developer.samsung.com/t/samsung-health-not-pushing-any-data-to-health-connect/23169
[28] https://forum.developer.samsung.com/t/hr-tag-measures-constantly-in-wear-os-4-bug/26545?page=2
[29] https://developer.samsung.com/health/sensor/faq.html
[30] https://developer.samsung.com/health/android/data/api-reference/com/samsung/android/sdk/healthdata/HealthConnectionErrorResult.html
[31] https://www.samsung.com/us/support/answer/ANS00091083/
[32] https://developer.samsung.com/health/sensor
[33] https://forum.developer.samsung.com/c/samsung-health/23?page=7
[34] https://xdaforums.com/t/restrictions-removed-samsung-health-monitor-wearos-phone-1-3-3-002-watch-1-3-2-025-root-age-country-restriction-removed-4th-sept-2024.4322527/
[35] https://developer.samsung.com/health/stack/release-notes/v11.html
[36] https://forum.developer.samsung.com/t/does-the-samsung-privileged-health-sdk-has-a-cost/28785

---
Answer from Perplexity: pplx.ai/share