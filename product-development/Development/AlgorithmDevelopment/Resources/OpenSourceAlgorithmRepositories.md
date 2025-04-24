https://github.com/ojwalch/sleep_accel
https://github.com/wadpac/Sundararajan-SleepClassification-2021




## Open Source Tools

Several open-source Python tools exist for developing and analyzing sleep tracking algorithms using wearable sensor data. These tools provide frameworks for data processing, visualization, and algorithm implementation, catering to both research and practical applications.

## **Key Python Libraries for Sleep Analysis**

### **1. Asleep**
- **Purpose**: Classifies sleep stages from wrist-worn accelerometer data using machine learning[3][11]
- **Features**:
  - Supports multiple wearable formats (AX3, ActiGraph, GENEActiv)
  - Outputs sleep duration, efficiency, and REM/NREM metrics
  - Includes visualization tools through `visu_sleep` command[3]
- **Usage**:
  ```bash
  pip install asleep
  get_sleep sample.cwa
  ```

### **2. SleepPy**
- **Focus**: Accelerometer-based sleep analysis with modular algorithms[5][17]
- **Capabilities**:
  - Calculates 35+ sleep metrics including WASO and sleep efficiency
  - Processes .gt3x, .bin, and CSV formats
  - Includes batch processing functionality[12]
- **Implementation**:
  ```python
  import sleeppy
  results = sleeppy.process(file_path)
  ```

### **3. pyActigraphy**
- **Scope**: Comprehensive actigraphy analysis toolbox[4][6][8]
- **Functionality**:
  - Multiple sleep detection algorithms (Cole-Kripke, Sadeh)
  - Rest-activity rhythm analysis (IS, IV, L5/L10 metrics)
  - Visualization of sleep agendas and circadian patterns[6]
- **Data Support**: Reads .awd, .mtn, and .csv formats[4]

### **4. YASA (Yet Another Spindle Algorithm)**
- **Strength**: Automated sleep staging with clinical-grade accuracy[7][9]
- **Technical Features**:
  - Pretrained neural networks for EEG/EMG analysis
  - Sleep spindle detection with automatic artifact rejection
  - Phase-amplitude coupling analysis[7]
- **Performance**: Matches human interscorer agreement (87.5% accuracy)[7]

### **5. SciKit Digital Health (SKDH)**
- **Framework**: End-to-end processing pipeline for wearable data[13]
- **Components**:
  - Gait analysis
  - Sit-to-stand detection
  - Sleep/wake classification
- **Design**: Modular architecture with automatic feature extraction[13]

## **Comparative Features**

| Tool          | Data Types       | ML Support | Visualization | Key Metrics Produced          |
|---------------|------------------|------------|---------------|-------------------------------|
| Asleep[3]     | Accelerometer    | ✓          | ✓             | Sleep duration, REM/NREM      |
| SleepPy[5]    | Accelerometer    | ✗          | ✓             | WASO, efficiency              |
| pyActigraphy[6]| Actigraphy       | ✗          | ✓             | IS, IV, L5                    |
| YASA[7]       | EEG/Acceleration | ✓          | ✓             | Sleep stages, spindles        |
| SKDH[13]      | Multi-modal      | ✓          | ✗             | Activity patterns, sleep/wake |

These tools enable researchers to:
1. Process raw accelerometer/actigraphy data[3][6]
2. Implement custom sleep scoring algorithms[5][13]
3. Validate against polysomnography references[7][11]
4. Generate publication-ready figures and statistics[1][6]

For wearable integration, **Asleep** and **SKDH** offer particularly streamlined workflows, with Asleep demonstrating validated performance on 1000+ nights of multi-center data[3]. YASA provides state-of-the-art neural network architectures for researchers interested in deep learning approaches[7][9]. All tools are available via PyPI or GitHub repositories with BSD-style licenses[3][5][7][9].

Citations:
[1] https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2017.00060/full
[2] https://pmc.ncbi.nlm.nih.gov/articles/PMC5613192/
[3] https://github.com/OxWearables/asleep
[4] https://pubmed.ncbi.nlm.nih.gov/34665807/
[5] https://www.pfirelab.com/publication-and-results/sleeppy-python-package-sleep-analysis-accelerometer-data
[6] https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1009514
[7] https://elifesciences.org/articles/70092
[8] https://orbi.uliege.be/bitstream/2268/267574/1/Hammad_2021_PlosComputBiol.pdf
[9] https://github.com/raphaelvallat/yasa
[10] https://www.reddit.com/r/Python/comments/1csa6j0/i_created_a_python_script_that_makes_it_easier_to/
[11] https://www.nature.com/articles/s41598-020-79294-y
[12] https://github.com/elyiorgos/sleeppy
[13] https://pmc.ncbi.nlm.nih.gov/articles/PMC9073613/
[14] https://stackoverflow.com/questions/42890925/producing-a-sleep-log-visualisation-in-python
[15] https://www.reddit.com/r/IOT/comments/116jo0z/smartband_or_fitness_tracker_that_provides_easy/
[16] https://www.youtube.com/watch?v=nSMt9nZRUdE
[17] https://joss.theoj.org/papers/10.21105/joss.01663
[18] https://www.researchgate.net/publication/355770176_pyActigraphy_Open-source_python_package_for_actigraphy_data_visualization_and_analysis

---
### Key Points
- Research suggests that the main open-source Python tools for sleep tracking using wearable data are Asleep, SleepPy, pyActigraphy, YASA, and SciKit Digital Health (SKDH), but they are primarily for post-processing, not real-time tracking.
- It seems likely that pyActigraphy has added support for light exposure analysis, enhancing its utility.
- The evidence leans toward SleepPy being no longer actively maintained, while Asleep, pyActigraphy, and SKDH are still maintained as of recent updates.

### Tool Overview
These tools help analyze sleep data from wearable devices like wristbands, focusing on processing and visualizing data after collection. They support various formats and metrics but aren't designed for real-time use, which might be unexpected for users seeking live tracking.

### Maintenance Status
- **Asleep**: Last updated in July 2024, actively maintained.
- **SleepPy**: Last updated in 2019, not actively maintained.
- **pyActigraphy**: Last updated in November 2023, with new features added.
- **YASA**: Last updated in December 2022, less recent activity.
- **SKDH**: Last updated in November 2024, actively maintained.

### Real-Time Consideration
It's worth noting that for real-time sleep tracking, these tools may not meet your needs, as they focus on post-analysis. You might need custom solutions for live data processing from wearables.

---

### Survey Note: Updated Report on Open-Source Python Tools for Sleep Tracking

This report updates the analysis of open-source Python libraries and tools for sleep tracking using data from wearable devices, focusing on their current status and suitability for real-time applications as of March 2025. The original report highlighted several key tools, and this update incorporates recent developments, maintenance status, and a clarification on real-time capabilities.

#### Introduction
Wearable devices, such as wrist-worn accelerometers and actigraphy sensors, have become vital for sleep research and personal health monitoring. Open-source Python tools facilitate the development and analysis of sleep tracking algorithms, providing frameworks for data processing, visualization, and algorithm implementation. This report reviews the primary tools—Asleep, SleepPy, pyActigraphy, YASA (Yet Another Spindle Algorithm), and SciKit Digital Health (SKDH)—and assesses their relevance, especially for real-time sleep tracking.

#### Key Python Libraries for Sleep Analysis

1. **Asleep**
   - **Purpose**: Designed to classify sleep stages from wrist-worn accelerometer data using machine learning, aligning with research needs for sleep stage analysis.
   - **Features**: Supports multiple wearable formats, including AX3, ActiGraph, and GENEActiv, and outputs metrics such as sleep duration, efficiency, and REM/NREM stages. Includes visualization tools via the `visu_sleep` command.
   - **Usage**: Installation is straightforward with `pip install asleep`, and usage examples include processing files like `get_sleep sample.cwa`.
   - **Status**: Actively maintained, with the last commit recorded in July 2024, indicating ongoing development and support.
   - **Relevance for Real-Time**: Primarily post-processing, not designed for real-time data streams from wearables.

2. **SleepPy**
   - **Focus**: Focuses on accelerometer-based sleep analysis, offering a modular framework for algorithm implementation.
   - **Capabilities**: Calculates over 35 sleep metrics, including WASO (Wake After Sleep Onset) and sleep efficiency, and processes formats like .gt3x, .bin, and CSV, with batch processing functionality.
   - **Implementation**: Can be used with `import sleeppy; results = sleeppy.process(file_path)`.
   - **Status**: Not actively maintained, with the last commit in August 2019, suggesting limited recent updates or community engagement.
   - **Relevance for Real-Time**: Similar to Asleep, it's designed for post-analysis, not real-time tracking.

3. **pyActigraphy**
   - **Scope**: A comprehensive toolbox for actigraphy data analysis, supporting various research and clinical applications.
   - **Functionality**: Includes multiple sleep detection algorithms (e.g., Cole-Kripke, Sadeh), rest-activity rhythm analysis (IS, IV, L5/L10 metrics), and visualization of sleep agendas and circadian patterns. Recent updates include support for light exposure analysis, enhancing its utility for comprehensive monitoring.
   - **Data Support**: Reads formats like .awd, .mtn, and .csv, with added support for new datasets from the National Sleep Research Resource.
   - **Status**: Maintained, with the last commit in November 2023, reflecting active development.
   - **Relevance for Real-Time**: Focused on post-processing, not real-time, but its expanded features make it versatile for detailed analysis.

4. **YASA (Yet Another Spindle Algorithm)**
   - **Strength**: Offers automated sleep staging with clinical-grade accuracy, particularly for EEG/EMG analysis.
   - **Technical Features**: Utilizes pretrained neural networks for sleep spindle detection, automatic artifact rejection, and phase-amplitude coupling analysis, achieving 87.5% accuracy compared to human interscorer agreement.
   - **Status**: Less recently updated, with the last commit in December 2022, but still relevant for specific research needs.
   - **Relevance for Real-Time**: Primarily for post-processing polysomnographic data, not suitable for real-time wearable data analysis.

5. **SciKit Digital Health (SKDH)**
   - **Framework**: Provides an end-to-end processing pipeline for wearable inertial sensor data, including sleep/wake classification.
   - **Components**: Covers gait analysis, sit-to-stand detection, and physical activity, with a modular architecture for feature extraction.
   - **Status**: Actively maintained, with the last commit in November 2024, indicating strong ongoing support.
   - **Relevance for Real-Time**: Designed for post-processing, not real-time, but its comprehensive approach is valuable for wearable data analysis.

#### Comparative Features

The following table summarizes the key attributes of each tool, updated to reflect recent developments:

| Tool          | Data Types       | ML Support | Visualization | Key Metrics Produced          | Maintenance Status       |
|---------------|------------------|------------|---------------|-------------------------------|--------------------------|
| Asleep[3]     | Accelerometer    | ✓          | ✓             | Sleep duration, REM/NREM      | Actively maintained      |
| SleepPy[5]    | Accelerometer    | ✗          | ✓             | WASO, efficiency              | Not actively maintained  |
| pyActigraphy[6]| Actigraphy       | ✗          | ✓             | IS, IV, L5, light exposure metrics | Maintained              |
| YASA[7]       | EEG/Acceleration | ✓          | ✓             | Sleep stages, spindles        | Less recently updated    |
| SKDH[13]      | Multi-modal      | ✓          | ✗             | Activity patterns, sleep/wake | Actively maintained      |

#### Discussion and Conclusion
These tools enable researchers to process raw accelerometer and actigraphy data, implement custom sleep scoring algorithms, validate against polysomnography references, and generate publication-ready figures and statistics. For wearable integration, Asleep and SKDH offer streamlined workflows, with Asleep validated on large datasets and SKDH providing a robust pipeline. YASA is notable for its deep learning approaches, particularly for EEG-based analysis.

However, it's crucial to note that none of these tools are designed for real-time sleep tracking, which may be a limitation for users seeking live monitoring from wearable devices. They are primarily for post-processing, analyzing data after collection. For real-time applications, users might need to explore custom solutions or integrate with streaming data frameworks, which are beyond the scope of these libraries.

The provided GitHub repositories (https://github.com/ojwalch/sleep_accel and https://github.com/wadpac/Sundararajan-SleepClassification-2021) were reviewed. The first is a Swift project for Apple Watch, not relevant for Python tools, and the second is a specific implementation for a research paper, not a general-purpose library, thus not included in this list.

#### Key Citations
- [Frontiers article on Sleep software](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2017.00060/full)
- [PMC article on Sleep software](https://pmc.ncbi.nlm.nih.gov/articles/PMC5613192/)
- [Asleep GitHub repository](https://github.com/OxWearables/asleep)
- [pyActigraphy PubMed reference](https://pubmed.ncbi.nlm.nih.gov/34665807/)
- [SleepPy publication and results](https://www.pfirelab.com/publication-and-results/sleeppy-python-package-sleep-analysis-accelerometer-data)
- [pyActigraphy PLOS article](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1009514)
- [YASA eLife article](https://elifesciences.org/articles/70092)
- [pyActigraphy additional reference](https://orbi.uliege.be/bitstream/2268/267574/1/Hammad_2021_PlosComputBiol.pdf)
- [YASA GitHub repository](https://github.com/raphaelvallat/yasa)
- [Reddit post on Python script](https://www.reddit.com/r/Python/comments/1csa6j0/i_created_a_python_script_that_makes_it_easier_to/)
- [Nature article on sleep research](https://www.nature.com/articles/s41598-020-79294-y)
- [SleepPy GitHub repository](https://github.com/elyiorgos/sleeppy)
- [SKDH PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9073613/)
- [Stack Overflow on sleep log visualization](https://stackoverflow.com/questions/42890925/producing-a-sleep-log-visualisation-in-python)
- [Reddit post on smartband](https://www.reddit.com/r/IOT/comments/116jo0z/smartband_or_fitness_tracker_that_provides_easy/)
- [YouTube video on sleep tracking](https://www.youtube.com/watch?v=nSMt9nZRUdE)
- [JOSS paper on SleepPy](https://joss.theoj.org/papers/10.21105/joss.01663)
- [ResearchGate on pyActigraphy](https://www.researchgate.net/publication/355770176_pyActigraphy_Open-source_python_package_for_actigraphy_data_visualization_and_analysis)