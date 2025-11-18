### Key Points
- Research suggests the most common features from accelerometer and PPG signals for sleep stage classification include movement intensity, heart rate, and heart rate variability.
- It seems likely that combining these features improves accuracy, especially for distinguishing between wake, light sleep, deep sleep, and REM.
- The evidence leans toward using machine learning algorithms like neural networks and random forests for classification.

---

### Features and Their Importance
Wearable sensors with accelerometers measure movement, helping identify wakefulness (more movement) versus sleep (less movement). Common features include activity counts and time since last movement, which are crucial for detecting periods of immobility typical in deeper sleep.

PPG sensors measure heart rate and its variability, capturing changes in the autonomic nervous system. Key features are heart rate, heart rate variability (e.g., RMSSD, LF/HF ratio), and breathing rate, which are vital for distinguishing sleep stages like deep sleep (high variability) and REM (lower variability).

### Synergy and Algorithms
Combining accelerometer and PPG features provides a fuller picture, with movement data complementing physiological data for better multi-stage classification. Algorithms like neural networks, random forests, and support vector machines are commonly used, trained on these features to classify sleep stages.

### How Features Are Derived
Features are extracted from continuous signals by sampling at high frequencies (e.g., 50 Hz for accelerometers, 32–64 Hz for PPG). Accelerometer data is processed into activity counts over 30-second epochs, while PPG data involves peak detection for heart rate and spectral analysis for variability, often after filtering and artifact removal.

---

---

### Survey Note: Detailed Analysis of Features from Wearable Sensors for Sleep Stage Classification

This note provides an in-depth examination of the most common features extracted from wearable sensors equipped with accelerometers and photoplethysmography (PPG) signals for sleep stage classification. It addresses the importance of these features, their synergy, the algorithms they are used with, and how they are derived from continuous time-domain input signals, based on recent research and literature.

#### Background and Context
Sleep stage classification traditionally relies on polysomnography (PSG), which is obtrusive and expensive. Wearable devices, leveraging accelerometer and PPG data, offer a portable, cost-effective alternative for out-of-lab sleep monitoring. Accelerometers measure movement, while PPG captures physiological signals like heart rate, enabling the classification of sleep stages such as wake, N1, N2, N3 (deep sleep), and REM. The combination of these sensors is increasingly adopted, as evidenced by studies published in recent years, such as [Evaluating reliability in wearable devices for sleep staging](https://www.nature.com/articles/s41746-024-01016-9) and [Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device](https://academic.oup.com/sleep/article/42/12/zsz180/5549536).

#### Most Common Features Extracted

The features extracted from accelerometer and PPG signals are critical for distinguishing between sleep stages. Below, we detail the most common features, categorized by sensor type:

##### Accelerometer Features
- **Activity Counts or Movement Intensity**: Computed over 30-second epochs by integrating the acceleration signal, often after low-pass filtering to remove noise. This captures total movement, which is higher during wakefulness and lower in sleep, particularly deep sleep.
- **Statistical Measures**: Include mean, maximum, minimum, and standard deviation of acceleration across each axis (x, y, z). These are calculated over epochs to quantify movement patterns, with lower values indicating deeper sleep stages.
- **Time Since Last Movement**: Determined by detecting significant acceleration events (e.g., above a threshold) and calculating the elapsed time, useful for identifying prolonged immobility characteristic of N3.

These features are derived from raw acceleration data, typically sampled at frequencies like 50 Hz, as noted in [Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device](https://academic.oup.com/sleep/article/42/12/zsz180/5549536).

##### PPG Features
- **Heart Rate (HR)**: Derived by detecting peaks in the PPG waveform, corresponding to heartbeats, and calculating the average beats per minute (BPM). HR is lower in deep sleep and more variable during REM.
- **Heart Rate Variability (HRV) Features**:
  - **Time-Domain Measures**: Include Root Mean Square of Successive Differences (RMSSD), Standard Deviation of NN intervals (SDNN), and percentage of successive RR intervals differing by more than 50 ms (pNN50). These measure short-term variability, higher in deep sleep.
  - **Frequency-Domain Measures**: Power in low-frequency (LF, 0.04–0.15 Hz), high-frequency (HF, 0.15–0.4 Hz), and very low-frequency (VLF, 0.015–0.04 Hz) bands, as well as the LF/HF ratio, reflecting autonomic balance.
- **Local Standard Deviation of Heart Rate**: Calculated over short windows (e.g., 30 seconds) to capture variability, useful for distinguishing sleep stages based on autonomic activity.
- **Breathing Rate**: Estimated from PPG by analyzing respiratory-induced variations, often using spectral analysis or zero-crossing methods, with more regular patterns in deep sleep.

These features are extracted after preprocessing, such as filtering and artifact rejection, as detailed in [A computationally efficient algorithm for wearable sleep staging in clinical populations](https://pmc.ncbi.nlm.nih.gov/articles/PMC10244431/). Additionally, studies like [Multi-stage sleep classification using photoplethysmographic sensor](https://pmc.ncbi.nlm.nih.gov/articles/PMC10090868/) highlight advanced PPG features, including surrogates for arterial blood pressure (e.g., areas under the PPG cycle) and nonlinear-dynamical features (e.g., fractal dimensions), though these are less common in standard wearables.

#### Importance of These Features

- **Accelerometer Features**: Movement is a primary indicator of wakefulness versus sleep, with wake stages showing higher activity counts and deeper sleep (N3) showing minimal movement. However, accelerometer data alone is limited for multi-stage classification, as noted in [Evaluating reliability in wearable devices for sleep staging](https://www.nature.com/articles/s41746-024-01016-9), where devices using only accelerometers are effective for sleep/wake detection but fall short for NREM/REM differentiation.
  
- **PPG Features**: These capture autonomic nervous system activity, which varies across sleep stages. For instance, HRV is higher in N3 due to parasympathetic dominance, while REM shows lower HRV and more variable heart rate, reflecting increased sympathetic activity. Breathing rate, derived from PPG, also varies, with more regular patterns in deep sleep, making it a valuable marker for stage classification.

#### Synergy Between Features

The synergy between accelerometer and PPG features lies in their complementary nature. Accelerometer data excels at detecting behavioral aspects like movement, which is crucial for binary sleep/wake classification, but struggles with finer distinctions between sleep stages. PPG data, on the other hand, provides physiological insights, particularly through HRV, enabling differentiation between N1, N2, N3, and REM. Studies show that combining these modalities significantly enhances accuracy, with [Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device](https://academic.oup.com/sleep/article/42/12/zsz180/5549536) reporting improved NREM/REM accuracy by 15%–25% when adding heart rate to motion features. This combination leverages both behavioral and physiological data, offering a more comprehensive view for multi-stage classification.

#### Algorithms Used for Sleep Stage Classification

The extracted features are typically fed into machine learning algorithms for classification. Common algorithms include:
- **Random Forests**: Effective for handling non-linear relationships, as seen in [Sleep classification from wrist-worn accelerometer data using random forests](https://www.nature.com/articles/s41598-020-79217-x), achieving F1 scores of 73.93% for sleep-wake classification.
- **Neural Networks**: Used in [A computationally efficient algorithm for wearable sleep staging in clinical populations](https://pmc.ncbi.nlm.nih.gov/articles/PMC10244431/), achieving a median accuracy of 77.8% for four-stage classification (wake, N1-2, N3, REM).
- **Support Vector Machines (SVM)**: Noted in [Multi-stage sleep classification using photoplethysmographic sensor](https://pmc.ncbi.nlm.nih.gov/articles/PMC10090868/), with polynomial kernels yielding accuracies up to 84.66% for two-stage classification.
- **Gradient Boosting Decision Trees (e.g., XGBoost)**: Mentioned in [A Multi-Level Classification Approach for Sleep Stage Prediction With Processed Data Derived From Consumer Wearable Activity Trackers](https://pmc.ncbi.nlm.nih.gov/articles/PMC8521802/), improving epoch-wise performance for sleep stage prediction.

These algorithms are trained on labeled PSG data and validated on independent datasets, ensuring robustness across different populations.

#### Derivation of Features from Continuous Time-Domain Signals

The process of deriving features from continuous signals involves several steps, ensuring accuracy and reliability:

##### Accelerometer
- **Raw Signal**: Typically sampled at high frequencies, such as 50 Hz for wrist-worn devices, as seen in [Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device](https://academic.oup.com/sleep/article/42/12/zsz180/5549536).
- **Preprocessing**: Includes low-pass filtering (e.g., 1 Hz cutoff) to remove high-frequency noise, as noted in [A computationally efficient algorithm for wearable sleep staging in clinical populations](https://pmc.ncbi.nlm.nih.gov/articles/PMC10244431/).
- **Feature Extraction**:
  - **Activity Counts**: Computed by integrating the acceleration signal over 30-second epochs, often involving summing absolute values minus gravity estimates.
  - **Statistical Measures**: Calculated as mean, max, min, and standard deviation for each axis over the epoch, capturing movement patterns.
  - **Time Since Last Movement**: Determined by detecting significant acceleration events (above a threshold) and calculating the elapsed time, useful for immobility analysis.

##### PPG
- **Raw Signal**: Sampled at frequencies like 32–64 Hz, as detailed in [A computationally efficient algorithm for wearable sleep staging in clinical populations](https://pmc.ncbi.nlm.nih.gov/articles/PMC10244431/).
- **Preprocessing**: Involves filtering (e.g., real-time moving average filters), artifact rejection (e.g., discarding data during motion artifacts >0.1 g), and interpolation for missing data, as seen in [Evaluating reliability in wearable devices for sleep staging](https://www.nature.com/articles/s41746-024-01016-9).
- **Feature Extraction**:
  - **Heart Rate (HR)**: Derived by detecting peaks in the PPG waveform, calculating inter-beat intervals (IBI), and computing HR as 60 / IBI.
  - **HRV Features**: 
    - **Time-Domain**: Computed from the IBI series, e.g., RMSSD is the square root of the mean of squared differences between successive IBIs.
    - **Frequency-Domain**: Involves applying a Fast Fourier Transform (FFT) to the IBI series to compute power in LF, HF, and VLF bands, as detailed in [Multi-stage sleep classification using photoplethysmographic sensor](https://pmc.ncbi.nlm.nih.gov/articles/PMC10090868/).
  - **Local Standard Deviation of HR**: Calculated over short windows (e.g., 30 seconds) to capture variability, as seen in [Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device](https://academic.oup.com/sleep/article/42/12/zsz180/5549536).
  - **Breathing Rate**: Estimated by analyzing periodic variations in the PPG signal, often using spectral analysis, with more regular patterns in deep sleep.

This detailed derivation ensures that features are robust and reflective of sleep stage characteristics, enhancing classification accuracy.

#### Performance and Validation
Studies report varying performance metrics, with [A computationally efficient algorithm for wearable sleep staging in clinical populations](https://pmc.ncbi.nlm.nih.gov/articles/PMC10244431/) achieving a median accuracy of 77.8% for four-stage classification, and [Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device](https://academic.oup.com/sleep/article/42/12/zsz180/5549536) reporting accuracies up to 80.1% for sleep-wake classification. These metrics highlight the effectiveness of combining accelerometer and PPG features, validated against PSG gold standards.

#### Unexpected Detail: Advanced PPG Features
While the most common features focus on HR and HRV, advanced studies like [Multi-stage sleep classification using photoplethysmographic sensor](https://pmc.ncbi.nlm.nih.gov/articles/PMC10090868/) extract 79 features from PPG, including surrogates for arterial blood pressure and nonlinear-dynamical features (e.g., fractal dimensions). These are less common in standard wearables but offer potential for future improvements, particularly for home-based monitoring.

#### Conclusion
The integration of accelerometer and PPG features provides a robust framework for sleep stage classification, leveraging movement and physiological data. Their derivation from continuous signals involves sophisticated preprocessing and analysis, supporting machine learning algorithms for accurate classification. This approach enhances accessibility and affordability, making sleep monitoring feasible outside clinical settings.

---

### Key Citations
- [Evaluating reliability in wearable devices for sleep staging](https://www.nature.com/articles/s41746-024-01016-9)
- [Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device](https://academic.oup.com/sleep/article/42/12/zsz180/5549536)
- [A computationally efficient algorithm for wearable sleep staging in clinical populations](https://pmc.ncbi.nlm.nih.gov/articles/PMC10244431/)
- [Multi-stage sleep classification using photoplethysmographic sensor](https://pmc.ncbi.nlm.nih.gov/articles/PMC10090868/)
- [Sleep classification from wrist-worn accelerometer data using random forests](https://www.nature.com/articles/s41598-020-79217-x)
- [A Multi-Level Classification Approach for Sleep Stage Prediction With Processed Data Derived From Consumer Wearable Activity Trackers](https://pmc.ncbi.nlm.nih.gov/articles/PMC8521802/)


---
# Most Common Features from Wearable Sensors for Sleep Stage Classification

Sleep staging is a complex process traditionally performed in clinical settings using polysomnography (PSG). However, the increasing availability of wearable sensors equipped with accelerometers and photoplethysmography (PPG) has opened new possibilities for non-invasive sleep monitoring. This report examines the most common features extracted from these sensors for sleep stage classification, their importance, synergistic relationships, associated algorithms, and derivation methods.

## Accelerometer-Derived Features

## Body Movement and Activity Parameters

Accelerometer-based activity metrics serve as the foundation for most wearable sleep analysis systems. These features primarily detect movement patterns that differentiate wake periods from sleep states.

**Derivation Process:** Raw 3-axis accelerometer signals (typically sampled at 30-100Hz) undergo several processing steps:

1. Filtering to remove noise and gravity components
    
2. Computing the Euclidean Norm Minus One (ENMO) or similar metrics
    
3. Aggregation into standard 30-second epochs using statistical measures[4](https://www.biorxiv.org/content/10.1101/2021.08.10.455812.full)
    

**Importance:** Activity-based features achieve high accuracy (86.7%) for binary sleep/wake classification but struggle with multi-stage classification[23](https://pmc.ncbi.nlm.nih.gov/articles/PMC10948771/). They form the basis of most commercial sleep trackers and serve as essential inputs to more complex algorithms.

**Common Features:**

- Activity counts (filtered sum of absolute acceleration values)
    
- Zero crossings (frequency of signal crossing the mean)
    
- Local range and variance of acceleration[4](https://www.biorxiv.org/content/10.1101/2021.08.10.455812.full)
    
- Motion cadence (rhythmicity of movements)
    

## Postural and Position Features

Body position and position changes provide valuable contextual information beyond basic movement detection.

**Derivation Process:** Position features are extracted by:

1. Calculating angles from gravitational components of accelerometer signals
    
2. Classifying into standard positions (supine, prone, side)
    
3. Detecting transitions between positions[2](https://pmc.ncbi.nlm.nih.gov/articles/PMC5656479/)[24](https://jcsm.aasm.org/doi/10.5664/jcsm.6802)
    

**Importance:** Studies show adults spend approximately 54.1% of time in bed in the side position, 37.5% on the back, and 7.3% in the front position, with an average of 1.6 position shifts per hour[16](https://www.dovepress.com/sleep-positions-and-nocturnal-body-movements-based-on-free-living-acce-peer-reviewed-fulltext-article-NSS). Position data significantly improves specificity of sleep/wake detection, addressing a common weakness in wearable sleep staging.

**Common Features:**

- Sleep posture classification (supine, prone, side)
    
- Position shift frequency and timing
    
- Duration in each position
    
- Postural transition characteristics
    

## Respiration-Related Features

Chest-worn accelerometers can capture subtle movements corresponding to breathing patterns.

**Derivation Process:** Respiratory features are derived by:

1. Isolating low-frequency components (0.1-0.5Hz) from the accelerometer signal
    
2. Applying bandpass filtering to extract respiratory rhythm
    
3. Computing respiratory rate, depth, and variability metrics[1](https://www.mdpi.com/1424-8220/24/17/5717)
    

**Importance:** These features enable detection of sleep-disordered breathing and contribute to differentiating sleep stages, particularly deep sleep (N3) from lighter stages[1](https://www.mdpi.com/1424-8220/24/17/5717). A chest-worn accelerometer approach achieved 80.8% accuracy with a Cohen's kappa of 0.68 for four-class sleep staging[1](https://www.mdpi.com/1424-8220/24/17/5717).

## PPG-Derived Features

## Heart Rate Variability Parameters

Heart rate variability (HRV) metrics derived from PPG are among the most discriminative features for sleep stage classification.

**Derivation Process:**

1. Detecting peaks in the PPG signal to identify pulse-to-pulse intervals
    
2. Computing time-domain, frequency-domain, and nonlinear metrics
    
3. Aggregating into 30-second epochs with potential overlap[9](https://pmc.ncbi.nlm.nih.gov/articles/PMC8923916/)
    

**Importance:** HRV parameters show distinct patterns across different sleep stages, reflecting autonomic nervous system modulation[7](https://pmc.ncbi.nlm.nih.gov/articles/PMC7658638/)[9](https://pmc.ncbi.nlm.nih.gov/articles/PMC8923916/). Research demonstrates HRV indices are non-stationary across sleep stage epochs, with notable shifts between REM episodes[9](https://pmc.ncbi.nlm.nih.gov/articles/PMC8923916/).

**Common Features:**

- Time-domain: SDNN, RMSSD, pNN50
    
- Frequency-domain: Low frequency (LF), high frequency (HF), and LF/HF ratio
    
- Nonlinear: Sample entropy, Poincaré plot descriptors
    

## Pulse Wave Characteristics

The morphology of PPG waveforms contains valuable information about vascular properties that change during different sleep stages.

**Derivation Process:**

1. Segmenting PPG signals into individual pulse waves
    
2. Extracting amplitude, width, and area metrics
    
3. Detecting PWA drops (sudden reductions in amplitude)[11](https://pmc.ncbi.nlm.nih.gov/articles/PMC7355400/)
    

**Importance:** PWA-drops have shown strong associations with cardiometabolic outcomes and vary significantly across sleep stages, with distinct patterns in N1, N2, N3, and REM sleep[11](https://pmc.ncbi.nlm.nih.gov/articles/PMC7355400/). Pulse Transit Time (PTT) demonstrates high sensitivity (91-94%) and specificity (95-97%) in differentiating types of respiratory events during sleep[10](https://pubmed.ncbi.nlm.nih.gov/9847267/).

**Common Features:**

- Pulse Wave Amplitude (PWA) and PWA drops
    
- Pulse Transit Time (PTT)
    
- Systolic/diastolic areas and their ratios
    
- Pulse wave rising time and decay time
    

## Blood Oxygen and Respiratory Parameters

PPG technology allows for extraction of both oxygen saturation and respiratory information.

**Derivation Process:**

1. Using red and infrared LED signals to calculate SpO2
    
2. Analyzing respiratory-induced intensity variations in the PPG signal
    
3. Extracting respiratory rate and pattern features[5](https://pmc.ncbi.nlm.nih.gov/articles/PMC10090868/)
    

**Importance:** The combination of oxygen and respiratory parameters enables detection of sleep-disordered breathing events and contributes to sleep stage discrimination, particularly between REM and NREM stages[6](https://pmc.ncbi.nlm.nih.gov/articles/PMC8122413/).

## Feature Processing and Fusion Approaches

## Multi-Level Fusion Methods

The integration of features from multiple sensors significantly enhances classification performance.

**Feature-Level Fusion:**  
Features from different modalities are combined before classification, creating a comprehensive input vector that leverages complementary information[18](https://pmc.ncbi.nlm.nih.gov/articles/PMC9550904/).

**Decision-Level Fusion:**  
Separate classifiers process different sensor modalities, with final decisions combined through voting, averaging, or weighted schemes[18](https://pmc.ncbi.nlm.nih.gov/articles/PMC9550904/).

**Multi-Level Fusion:**  
This approach uses both feature-level and decision-level fusion, first combining extracted features and then blending classification outputs, achieving superior performance (87.2% accuracy) compared to single-level approaches[18](https://pmc.ncbi.nlm.nih.gov/articles/PMC9550904/)[19](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.727139/full).

**Importance:** Studies demonstrate that multi-level fusion particularly improves classification of challenging sleep stages like N1, which typically has the lowest performance in automated systems[18](https://pmc.ncbi.nlm.nih.gov/articles/PMC9550904/)[19](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.727139/full).

## Temporal Context Integration

Sleep is a temporally structured process, making context from surrounding epochs crucial for accurate classification.

**Derivation Process:**

1. Creating sliding windows encompassing multiple epochs
    
2. Computing features that capture transitions between states
    
3. Incorporating long-range dependencies through recurrent processing[4](https://www.biorxiv.org/content/10.1101/2021.08.10.455812.full)[7](https://pmc.ncbi.nlm.nih.gov/articles/PMC7658638/)
    

**Importance:** Research indicates that prior 250 seconds of movements are highly informative for current sleep stage classification[4](https://www.biorxiv.org/content/10.1101/2021.08.10.455812.full). Temporal context integration improves staging consistency and reduces implausible transitions.

## Classification Algorithms

## Traditional Machine Learning Approaches

Conventional algorithms remain competitive when paired with well-engineered features.

**Support Vector Machine (SVM):**  
Particularly effective with PPG-derived features, achieving accuracies of 84.66%, 79.62%, and 72.23% for two-, three-, and four-stage sleep classification using polynomial kernels[5](https://pmc.ncbi.nlm.nih.gov/articles/PMC10090868/).

**Random Forest:**  
Performs well with heterogeneous feature sets and provides feature importance rankings that aid in understanding sleep physiology[13](https://arxiv.org/abs/2303.02467).

**Importance:** These algorithms offer interpretability and computational efficiency, making them suitable for deployment on resource-constrained wearable devices.

## Deep Learning Methods

End-to-end deep learning approaches have demonstrated significant advances in recent years.

**Convolutional-Recurrent Architectures:**  
Combine CNN layers for feature extraction with LSTM or GRU layers for temporal modeling, achieving 80.1% accuracy (κ = 0.65) for three-stage classification directly from raw PPG signals[7](https://pmc.ncbi.nlm.nih.gov/articles/PMC7658638/).

**Specialized Networks:**  
Purpose-built architectures like SleepPPG-Net use residual convolutional networks and temporal convolutional networks for robust feature extraction from raw PPG signals[20](https://pubmed.ncbi.nlm.nih.gov/36446010/)[21](https://peterhcharlton.github.io/publication/kotzen-2022/).

**Importance:** These models approach the accuracy of human scorers (80-85% accuracy, κ = 0.7-0.75) for four-stage classification and eliminate the need for manual feature engineering[22](https://research.google/pubs/sleep-staging-classification-from-wearable-signals-using-deep-learning/).

## Synergistic Relationships Between Features

The most effective sleep staging systems exploit complementary relationships between different feature types:

1. **Accelerometer and PPG Complementarity:**  
    Accelerometer data excels at detecting wake states and body movements, while PPG features better distinguish between sleep stages (particularly REM vs. NREM), leading to a trend toward combining these sensors in modern wearables[23](https://pmc.ncbi.nlm.nih.gov/articles/PMC10948771/).
    
2. **Position and HRV Interactions:**  
    Sleep position affects autonomic nervous system activity, creating position-dependent changes in HRV parameters that must be accounted for in classification algorithms[2](https://pmc.ncbi.nlm.nih.gov/articles/PMC5656479/)[8](https://pmc.ncbi.nlm.nih.gov/articles/PMC10566244/).
    
3. **Respiratory and Cardiac Feature Synergy:**  
    Cardiorespiratory coupling varies across sleep stages, making the combination of these feature domains particularly powerful for distinguishing N3 (deep) sleep from lighter stages[1](https://www.mdpi.com/1424-8220/24/17/5717)[6](https://pmc.ncbi.nlm.nih.gov/articles/PMC8122413/).
    
4. **Short-term and Long-term Temporal Features:**  
    Combining features that capture immediate physiological state with those that reflect longer-term patterns enhances classification robustness, particularly during transitional periods between sleep stages[9](https://pmc.ncbi.nlm.nih.gov/articles/PMC8923916/).
    

## Conclusion

The integration of accelerometer and PPG sensors in wearable devices has enabled increasingly sophisticated sleep stage classification outside laboratory settings. The most discriminative features combine movement patterns, posture information, cardiac variability, pulse wave characteristics, and respiratory parameters through multi-level fusion approaches. While traditional machine learning with expertly crafted features remains effective, end-to-end deep learning methods are approaching the performance of human experts.

Current research suggests that no single feature type is sufficient for reliable multi-stage sleep classification. The trend toward combining complementary sensors and advanced fusion techniques reflects the complex, multi-system nature of sleep physiology and represents the most promising direction for wearable sleep monitoring technologies[23](https://pmc.ncbi.nlm.nih.gov/articles/PMC10948771/).

### Citations:

1. [https://www.mdpi.com/1424-8220/24/17/5717](https://www.mdpi.com/1424-8220/24/17/5717)
2. [https://pmc.ncbi.nlm.nih.gov/articles/PMC5656479/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5656479/)
3. [https://pmc.ncbi.nlm.nih.gov/articles/PMC7956647/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7956647/)
4. [https://www.biorxiv.org/content/10.1101/2021.08.10.455812.full](https://www.biorxiv.org/content/10.1101/2021.08.10.455812.full)
5. [https://pmc.ncbi.nlm.nih.gov/articles/PMC10090868/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10090868/)
6. [https://pmc.ncbi.nlm.nih.gov/articles/PMC8122413/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8122413/)
7. [https://pmc.ncbi.nlm.nih.gov/articles/PMC7658638/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7658638/)
8. [https://pmc.ncbi.nlm.nih.gov/articles/PMC10566244/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10566244/)
9. [https://pmc.ncbi.nlm.nih.gov/articles/PMC8923916/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8923916/)
10. [https://pubmed.ncbi.nlm.nih.gov/9847267/](https://pubmed.ncbi.nlm.nih.gov/9847267/)
11. [https://pmc.ncbi.nlm.nih.gov/articles/PMC7355400/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7355400/)
12. [https://pmc.ncbi.nlm.nih.gov/articles/PMC5568624/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5568624/)
13. [https://arxiv.org/abs/2303.02467](https://arxiv.org/abs/2303.02467)
14. [https://www.medrxiv.org/content/10.1101/2023.11.03.23297889v1.full.pdf](https://www.medrxiv.org/content/10.1101/2023.11.03.23297889v1.full.pdf)
15. [https://www.apple.com/healthcare/docs/site/Estimating_Sleep_Stages_from_Apple_Watch_Sept_2023.pdf](https://www.apple.com/healthcare/docs/site/Estimating_Sleep_Stages_from_Apple_Watch_Sept_2023.pdf)
16. [https://www.dovepress.com/sleep-positions-and-nocturnal-body-movements-based-on-free-living-acce-peer-reviewed-fulltext-article-NSS](https://www.dovepress.com/sleep-positions-and-nocturnal-body-movements-based-on-free-living-acce-peer-reviewed-fulltext-article-NSS)
17. [https://arxiv.org/abs/2111.10245](https://arxiv.org/abs/2111.10245)
18. [https://pmc.ncbi.nlm.nih.gov/articles/PMC9550904/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9550904/)
19. [https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.727139/full](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.727139/full)
20. [https://pubmed.ncbi.nlm.nih.gov/36446010/](https://pubmed.ncbi.nlm.nih.gov/36446010/)
21. [https://peterhcharlton.github.io/publication/kotzen-2022/](https://peterhcharlton.github.io/publication/kotzen-2022/)
22. [https://research.google/pubs/sleep-staging-classification-from-wearable-signals-using-deep-learning/](https://research.google/pubs/sleep-staging-classification-from-wearable-signals-using-deep-learning/)
23. [https://pmc.ncbi.nlm.nih.gov/articles/PMC10948771/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10948771/)
24. [https://jcsm.aasm.org/doi/10.5664/jcsm.6802](https://jcsm.aasm.org/doi/10.5664/jcsm.6802)
25. [https://pmc.ncbi.nlm.nih.gov/articles/PMC9708010/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9708010/)
26. [https://pmc.ncbi.nlm.nih.gov/articles/PMC9869419/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9869419/)
27. [https://www.nature.com/articles/s41598-023-36444-2](https://www.nature.com/articles/s41598-023-36444-2)
28. [https://www.reviewed.com/sleep/features/types-sleep-trackers-and-how-they-work](https://www.reviewed.com/sleep/features/types-sleep-trackers-and-how-they-work)
29. [https://www.nature.com/articles/s41598-024-75935-8](https://www.nature.com/articles/s41598-024-75935-8)
30. [https://pmc.ncbi.nlm.nih.gov/articles/PMC7191872/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7191872/)
31. [https://www.nature.com/articles/s41746-019-0210-1](https://www.nature.com/articles/s41746-019-0210-1)
32. [https://www.biorxiv.org/content/10.1101/2023.10.05.561041v1.full-text](https://www.biorxiv.org/content/10.1101/2023.10.05.561041v1.full-text)
33. [https://www.nature.com/articles/s41746-024-01016-9](https://www.nature.com/articles/s41746-024-01016-9)
34. [https://pubmed.ncbi.nlm.nih.gov/22256128/](https://pubmed.ncbi.nlm.nih.gov/22256128/)
35. [https://www.youtube.com/watch?v=YTbkq3E_Xgw](https://www.youtube.com/watch?v=YTbkq3E_Xgw)
36. [https://www.sciencedirect.com/science/article/abs/pii/S138994572400248X](https://www.sciencedirect.com/science/article/abs/pii/S138994572400248X)
37. [https://pmc.ncbi.nlm.nih.gov/articles/PMC10350137/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10350137/)
38. [https://pmc.ncbi.nlm.nih.gov/articles/PMC8113882/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8113882/)
39. [https://www.sciencedirect.com/science/article/pii/S2405844024078708](https://www.sciencedirect.com/science/article/pii/S2405844024078708)
40. [https://www.sleepfoundation.org/sleep-studies/actigraphy](https://www.sleepfoundation.org/sleep-studies/actigraphy)
41. [https://arxiv.org/pdf/2405.11008.pdf](https://arxiv.org/pdf/2405.11008.pdf)
42. [https://ojs.aaai.org/index.php/AAAI-SS/article/download/31246/33406/35302](https://ojs.aaai.org/index.php/AAAI-SS/article/download/31246/33406/35302)
43. [https://www.nature.com/articles/s41746-024-01065-0](https://www.nature.com/articles/s41746-024-01065-0)
44. [https://www.nature.com/articles/s41598-020-79217-x](https://www.nature.com/articles/s41598-020-79217-x)
45. [https://pmc.ncbi.nlm.nih.gov/articles/PMC7595622/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7595622/)
46. [https://pmc.ncbi.nlm.nih.gov/articles/PMC6942683/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6942683/)
47. [https://pmc.ncbi.nlm.nih.gov/articles/PMC6930135/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6930135/)
48. [https://www.medrxiv.org/content/10.1101/2022.03.07.22270992v1.full-text](https://www.medrxiv.org/content/10.1101/2022.03.07.22270992v1.full-text)
49. [https://pmc.ncbi.nlm.nih.gov/articles/PMC7867075/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7867075/)
50. [https://pmc.ncbi.nlm.nih.gov/articles/PMC6113241/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6113241/)
51. [https://techfinder.stanford.edu/technology/automated-classification-sleep-and-wake-single-day-triaxial-accelerometer-data](https://techfinder.stanford.edu/technology/automated-classification-sleep-and-wake-single-day-triaxial-accelerometer-data)
52. [https://www.sciencedirect.com/science/article/abs/pii/S2352721815001473](https://www.sciencedirect.com/science/article/abs/pii/S2352721815001473)
53. [https://rogersgroup.northwestern.edu/files/2019/sleepnpj.pdf](https://rogersgroup.northwestern.edu/files/2019/sleepnpj.pdf)
54. [https://dl.acm.org/doi/10.1109/ICCCN.2009.5235366](https://dl.acm.org/doi/10.1109/ICCCN.2009.5235366)
55. [https://mhealth.jmir.org/2021/10/e29849](https://mhealth.jmir.org/2021/10/e29849)
56. [https://www.labfront.com/course-video/traditional-signal-processing-sleep-wake-classification](https://www.labfront.com/course-video/traditional-signal-processing-sleep-wake-classification)
57. [https://royalsocietypublishing.org/doi/abs/10.1098/rsos.221517](https://royalsocietypublishing.org/doi/abs/10.1098/rsos.221517)
58. [https://peterhcharlton.github.io/publication/wearable_ppg_chapter/Wear_PPG_Chapter_20210323.pdf](https://peterhcharlton.github.io/publication/wearable_ppg_chapter/Wear_PPG_Chapter_20210323.pdf)
59. [https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0297582](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0297582)
60. [https://www.viatomtech.com/post/how-ppg-technology-reforms-sleep-studies-and-aids-in-sleep-apnea-diagnosis](https://www.viatomtech.com/post/how-ppg-technology-reforms-sleep-studies-and-aids-in-sleep-apnea-diagnosis)
61. [https://pmc.ncbi.nlm.nih.gov/articles/PMC10686289/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10686289/)
62. [https://www.nature.com/articles/s41746-020-0291-x](https://www.nature.com/articles/s41746-020-0291-x)
63. [https://jcsm.aasm.org/doi/10.5664/jcsm.10300](https://jcsm.aasm.org/doi/10.5664/jcsm.10300)
64. [https://arxiv.org/html/2410.00693](https://arxiv.org/html/2410.00693)
65. [https://www.mdpi.com/2079-9292/12/13/2923](https://www.mdpi.com/2079-9292/12/13/2923)
66. [https://onlinelibrary.wiley.com/doi/full/10.1002/aisy.202300270](https://onlinelibrary.wiley.com/doi/full/10.1002/aisy.202300270)
67. [https://www.nature.com/articles/s41746-021-00510-8](https://www.nature.com/articles/s41746-021-00510-8)
68. [https://jcsm.aasm.org/doi/full/10.5664/jcsm.10690](https://jcsm.aasm.org/doi/full/10.5664/jcsm.10690)
69. [https://pmc.ncbi.nlm.nih.gov/articles/PMC8253894/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8253894/)
70. [https://www.sciencedirect.com/science/article/pii/S2589004221004296](https://www.sciencedirect.com/science/article/pii/S2589004221004296)
71. [https://www.sciencedirect.com/science/article/pii/S1087079224000017](https://www.sciencedirect.com/science/article/pii/S1087079224000017)
72. [https://www.nature.com/articles/s41598-019-49703-y](https://www.nature.com/articles/s41598-019-49703-y)
73. [https://pmc.ncbi.nlm.nih.gov/articles/PMC9584568/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9584568/)
74. [https://www.cl.cam.ac.uk/teaching/2324/MH/MH-lecture4.pdf](https://www.cl.cam.ac.uk/teaching/2324/MH/MH-lecture4.pdf)
75. [https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2017.01100/full](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2017.01100/full)
76. [https://royalsocietypublishing.org/doi/10.1098/rsos.221517](https://royalsocietypublishing.org/doi/10.1098/rsos.221517)
77. [https://www.biorxiv.org/content/10.1101/2022.05.13.491872v1.full](https://www.biorxiv.org/content/10.1101/2022.05.13.491872v1.full)
78. [https://www.mdpi.com/1424-8220/23/22/9077](https://www.mdpi.com/1424-8220/23/22/9077)
79. [https://pmc.ncbi.nlm.nih.gov/articles/PMC5624990/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5624990/)
80. [https://academic.oup.com/sleep/article/43/11/zsaa098/5841624](https://academic.oup.com/sleep/article/43/11/zsaa098/5841624)
81. [https://www.sciencedirect.com/science/article/abs/pii/S0010482517303244](https://www.sciencedirect.com/science/article/abs/pii/S0010482517303244)
82. [https://www.sciencedirect.com/science/article/pii/S2666990022000027](https://www.sciencedirect.com/science/article/pii/S2666990022000027)
83. [https://www.sciencedirect.com/science/article/pii/S2589750020302466](https://www.sciencedirect.com/science/article/pii/S2589750020302466)
84. [https://www.neurologylive.com/view/pulse-wave-amplitude-respiratory-events](https://www.neurologylive.com/view/pulse-wave-amplitude-respiratory-events)
85. [https://www.medrxiv.org/content/10.1101/2023.07.25.23293134v1.full-text](https://www.medrxiv.org/content/10.1101/2023.07.25.23293134v1.full-text)
86. [https://jcsm.aasm.org/doi/10.5664/jcsm.11112](https://jcsm.aasm.org/doi/10.5664/jcsm.11112)
87. [https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2018.00948/full](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2018.00948/full)
88. [https://pubmed.ncbi.nlm.nih.gov/27993286/](https://pubmed.ncbi.nlm.nih.gov/27993286/)
89. [https://www.sciencedirect.com/science/article/pii/S0167527323007398](https://www.sciencedirect.com/science/article/pii/S0167527323007398)
90. [https://pubmed.ncbi.nlm.nih.gov/29059825/](https://pubmed.ncbi.nlm.nih.gov/29059825/)
91. [https://www.mdpi.com/1424-8220/23/18/7931](https://www.mdpi.com/1424-8220/23/18/7931)
92. [https://www.sciencedirect.com/science/article/pii/S0954611116303092](https://www.sciencedirect.com/science/article/pii/S0954611116303092)
93. [https://www.sciencedirect.com/science/article/abs/pii/S1389945720300186](https://www.sciencedirect.com/science/article/abs/pii/S1389945720300186)
94. [https://pmc.ncbi.nlm.nih.gov/articles/PMC9056464/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9056464/)
95. [https://www.nature.com/articles/s41598-021-01358-4](https://www.nature.com/articles/s41598-021-01358-4)
96. [https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2022.1092222/pdf](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2022.1092222/pdf)
97. [https://www.sleepfoundation.org/best-sleep-trackers](https://www.sleepfoundation.org/best-sleep-trackers)
98. [https://pmc.ncbi.nlm.nih.gov/articles/PMC10452545/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10452545/)
99. [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1224784/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1224784/full)
100. [https://pmc.ncbi.nlm.nih.gov/articles/PMC8161815/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8161815/)
101. [https://pmc.ncbi.nlm.nih.gov/articles/PMC6579636/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6579636/)
102. [https://www.nature.com/articles/s41598-023-45020-7](https://www.nature.com/articles/s41598-023-45020-7)
103. [https://www.mdpi.com/1424-8220/20/15/4323](https://www.mdpi.com/1424-8220/20/15/4323)
104. [https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2021.665946/full](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2021.665946/full)
105. [https://academic.oup.com/sleep/article/47/4/zsad325/7501518](https://academic.oup.com/sleep/article/47/4/zsad325/7501518)
106. [https://datascience.stackexchange.com/questions/51865/calculate-a-ranking-function-from-classification-features](https://datascience.stackexchange.com/questions/51865/calculate-a-ranking-function-from-classification-features)
107. [https://www.sciencedirect.com/science/article/abs/pii/S0169260721001668](https://www.sciencedirect.com/science/article/abs/pii/S0169260721001668)
108. [https://respiratory-therapy.com/disorders-diseases/chronic-pulmonary-disorders/asthma/body-position-and-osa/](https://respiratory-therapy.com/disorders-diseases/chronic-pulmonary-disorders/asthma/body-position-and-osa/)
109. [https://web.fibion.com/articles/actigraphy-sleep-patterns-analysis/](https://web.fibion.com/articles/actigraphy-sleep-patterns-analysis/)
110. [https://jcsm.aasm.org/doi/10.5664/jcsm.7932](https://jcsm.aasm.org/doi/10.5664/jcsm.7932)
111. [https://www.sciencedirect.com/science/article/pii/S0952197624019171](https://www.sciencedirect.com/science/article/pii/S0952197624019171)
112. [https://www.cs.virginia.edu/~stankovic/psfiles/WISP_Sleeping_08_06.pdf](https://www.cs.virginia.edu/~stankovic/psfiles/WISP_Sleeping_08_06.pdf)
113. [https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2021.721919/full](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2021.721919/full)
114. [https://www.frontiersin.org/articles/10.3389/fpubh.2022.1092222/full](https://www.frontiersin.org/articles/10.3389/fpubh.2022.1092222/full)
115. [https://pmc.ncbi.nlm.nih.gov/articles/PMC8436137/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8436137/)
116. [https://aasm.org/staying-current-with-actigraphy-devices-for-sleep-wake-monitoring/](https://aasm.org/staying-current-with-actigraphy-devices-for-sleep-wake-monitoring/)
117. [https://pmc.ncbi.nlm.nih.gov/articles/PMC5781106/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5781106/)
118. [https://pubmed.ncbi.nlm.nih.gov/36238370/](https://pubmed.ncbi.nlm.nih.gov/36238370/)
119. [https://arxiv.org/pdf/2306.15808.pdf](https://arxiv.org/pdf/2306.15808.pdf)
120. [http://ml4h.cc/2019/pdf/125_ml4h_preprint.pdf](http://ml4h.cc/2019/pdf/125_ml4h_preprint.pdf)
121. [https://pubmed.ncbi.nlm.nih.gov/38959148/](https://pubmed.ncbi.nlm.nih.gov/38959148/)
122. [https://www.mdpi.com/2073-431X/13/1/13](https://www.mdpi.com/2073-431X/13/1/13)
123. [https://www.mdpi.com/2072-4292/15/17/4148](https://www.mdpi.com/2072-4292/15/17/4148)
124. [https://www.sciencedirect.com/science/article/pii/S1389945722011327](https://www.sciencedirect.com/science/article/pii/S1389945722011327)
125. [https://www.nature.com/articles/s41746-024-01086-9](https://www.nature.com/articles/s41746-024-01086-9)
126. [https://dl.acm.org/doi/full/10.1145/3543848](https://dl.acm.org/doi/full/10.1145/3543848)
127. [https://www.sciencedirect.com/science/article/abs/pii/S0169260722004941](https://www.sciencedirect.com/science/article/abs/pii/S0169260722004941)
128. [https://pmc.ncbi.nlm.nih.gov/articles/PMC9364961/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9364961/)
129. [https://pmc.ncbi.nlm.nih.gov/articles/PMC11326349/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11326349/)
130. [https://researchoutput.csu.edu.au/files/533634247/533600795_Published_article.pdf](https://researchoutput.csu.edu.au/files/533634247/533600795_Published_article.pdf)
131. [https://www.nature.com/articles/s41598-024-76197-0](https://www.nature.com/articles/s41598-024-76197-0)
132. [https://arxiv.org/abs/2404.06869](https://arxiv.org/abs/2404.06869)
133. [https://arxiv.org/pdf/2202.05735.pdf](https://arxiv.org/pdf/2202.05735.pdf)
134. [https://pmc.ncbi.nlm.nih.gov/articles/PMC10400763/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10400763/)
135. [https://arxiv.org/html/2502.17486v1](https://arxiv.org/html/2502.17486v1)
136. [https://www.sciencedirect.com/science/article/abs/pii/S0010482523006583](https://www.sciencedirect.com/science/article/abs/pii/S0010482523006583)
137. [https://paperswithcode.com/task/sleep-stage-detection](https://paperswithcode.com/task/sleep-stage-detection)
138. [https://www.sciencedirect.com/science/article/abs/pii/S1877750321002003](https://www.sciencedirect.com/science/article/abs/pii/S1877750321002003)
139. [https://pmc.ncbi.nlm.nih.gov/articles/PMC10654909/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10654909/)
140. [https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2019.00958/full](https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2019.00958/full)
141. [https://www.nature.com/articles/s41598-024-54817-z](https://www.nature.com/articles/s41598-024-54817-z)
142. [https://www.generalsleep.com/zmachine-synergy.html](https://www.generalsleep.com/zmachine-synergy.html)
143. [https://onlinelibrary.wiley.com/doi/10.1111/jsr.14143](https://onlinelibrary.wiley.com/doi/10.1111/jsr.14143)
144. [https://pmc.ncbi.nlm.nih.gov/articles/PMC10039037/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10039037/)
145. [https://beacon.bio/dreem-headband/](https://beacon.bio/dreem-headband/)
146. [https://www.sciencedirect.com/science/article/abs/pii/S1087079219301959](https://www.sciencedirect.com/science/article/abs/pii/S1087079219301959)
147. [https://pmc.ncbi.nlm.nih.gov/articles/PMC8780755/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8780755/)

---

Answer from Perplexity: [pplx.ai/share](https://www.perplexity.ai/search/pplx.ai/share)