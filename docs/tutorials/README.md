# Tutorials and User Guides

Welcome to the Adaptive Sleep Algorithms tutorials! These guides will help you get started with sleep signal analysis and master advanced techniques.

## 📚 Tutorial Path

### For Beginners

1. **[Getting Started](getting-started.md)** ⭐ START HERE
   - Installation and setup
   - Your first workflow
   - Understanding basic concepts
   - Quick start example

2. **[Common Workflows](common-workflows.md)**
   - HRV analysis
   - Movement detection
   - Multi-sensor feature extraction
   - Sleep staging preparation
   - Correlation analysis

### For All Users

3. **[Feature Extraction Guide](feature-extraction-guide.md)**
   - Statistical features
   - HRV features
   - Movement features
   - Correlation features
   - Sleep stage features
   - Custom features

4. **[Best Practices](best-practices.md)**
   - Data organization
   - Workflow design
   - Quality control
   - Performance optimization
   - Reproducibility
   - Common pitfalls

### For Developers

5. **[Python API Guide](python-api-guide.md)**
   - Programmatic usage
   - Working with signals
   - Feature extraction in code
   - Integration with existing scripts
   - Jupyter notebook examples

---

## 🎯 Quick Navigation by Task

### I want to...

**...analyze heart rate variability (HRV)**
→ [Common Workflows: HRV Analysis](common-workflows.md#1-heart-rate-variability-hrv-analysis)

**...detect movement and activity**
→ [Common Workflows: Movement Detection](common-workflows.md#2-movement-and-activity-detection)

**...extract features from multiple sensors**
→ [Common Workflows: Multi-Sensor Features](common-workflows.md#3-multi-sensor-feature-extraction)

**...prepare data for sleep staging**
→ [Common Workflows: Sleep Staging Prep](common-workflows.md#4-sleep-staging-preparation)

**...understand all available features**
→ [Feature Extraction Guide](feature-extraction-guide.md)

**...use Python instead of YAML**
→ [Python API Guide](python-api-guide.md)

**...improve my workflow quality**
→ [Best Practices](best-practices.md)

**...troubleshoot an error**
→ [Troubleshooting Guide](../troubleshooting.md)

---

## 📖 Documentation Overview

### User Documentation (you are here)

- **tutorials/** - Step-by-step guides (this folder)
- **[data-preparation.md](../data-preparation.md)** - How to prepare and import data
- **[troubleshooting.md](../troubleshooting.md)** - Common issues and solutions
- **[feature-extraction-examples.md](../feature-extraction-examples.md)** - Detailed examples

### Technical Documentation

- **requirements/** - Functional and technical requirements
- **designs/** - Architecture and design documents
- **diagrams/** - System architecture diagrams
- **[coding_guidelines.md](../coding_guidelines.md)** - For contributors

---

## 🚀 Quickstart

### 1-Minute Start

```bash
# Install the framework
git clone https://github.com/yourusername/adaptive-sleep-algorithms.git
cd adaptive-sleep-algorithms
pip install -e .

# Run an example workflow
python -m sleep_analysis.cli.run_workflow \
  --workflow workflows/complete_sleep_analysis.yaml \
  --data-dir data
```

### 5-Minute Tutorial

**Step 1: Create a simple workflow file (`my_workflow.yaml`):**

```yaml
collection_settings:
  epoch_grid_config:
    window_length: "30s"
    step_size: "30s"

import:
  - signal_type: "heart_rate"
    importer: "MergingImporter"
    source: "."
    config:
      file_pattern: "Polar_H10_*_HR.txt"
      time_column: "Phone timestamp"
      sort_by: "timestamp"
      delimiter: ";"
    sensor_type: "EKG"
    sensor_model: "PolarH10"
    body_position: "chest"
    base_name: "hr"

steps:
  - type: collection
    operation: "generate_epoch_grid"
  - type: multi_signal
    operation: "compute_hrv_features"
    inputs: ["hr"]
    parameters:
      hrv_metrics: ["hr_mean", "hr_std", "hr_cv", "hr_range"]
      use_rr_intervals: false
    output: "hrv"

export:
  - formats: ["csv", "excel"]
    output_dir: "results"
    content: ["hrv"]
```

**Step 2: Run it:**

```bash
python -m sleep_analysis.cli.run_workflow \
  --workflow my_workflow.yaml \
  --data-dir data
```

**Step 3: Check results:**

```bash
ls results/
# hrv.csv
# hrv.xlsx
```

---

## 💡 Learning Paths

### Path 1: Research User (YAML Workflows)

1. [Getting Started](getting-started.md) - Understand the basics
2. [Common Workflows](common-workflows.md) - Copy and adapt examples
3. [Feature Extraction Guide](feature-extraction-guide.md) - Learn available features
4. [Best Practices](best-practices.md) - Production-quality analysis

**Time:** 2-3 hours to proficiency

### Path 2: Developer (Python API)

1. [Getting Started](getting-started.md) - Understand the framework
2. [Python API Guide](python-api-guide.md) - Learn programmatic usage
3. [Feature Extraction Guide](feature-extraction-guide.md) - Available operations
4. [Best Practices](best-practices.md) - Code quality tips

**Time:** 3-4 hours to proficiency

### Path 3: Quick Analysis

1. [Getting Started: Quick Start](getting-started.md#quick-start-your-first-workflow) - 10 minutes
2. [Common Workflows](common-workflows.md) - Pick your analysis - 20 minutes
3. Run and iterate - Ongoing

**Time:** 30 minutes to first results

---

## 🎓 Example Workflows

The framework includes several ready-to-use example workflows in `workflows/`:

| Workflow | Description | Complexity |
|----------|-------------|------------|
| `polar_workflow.yaml` | Import Polar sensor data | Beginner |
| `complete_sleep_analysis.yaml` | Full feature extraction pipeline | Intermediate |
| `sleep_staging_with_rf.yaml` | Apply trained sleep staging model | Intermediate |
| `train_sleep_staging_model.yaml` | Train sleep staging model | Advanced |

### Try Them

```bash
# Simple import
python -m sleep_analysis.cli.run_workflow \
  --workflow workflows/polar_workflow.yaml \
  --data-dir data

# Complete analysis
python -m sleep_analysis.cli.run_workflow \
  --workflow workflows/complete_sleep_analysis.yaml \
  --data-dir data
```

---

## 📊 Use Case Examples

### Sleep Study Analysis

**Goal:** Extract comprehensive features for sleep staging

**Workflow:** [Sleep Staging Preparation](common-workflows.md#4-sleep-staging-preparation)

**Features Extracted:**
- HRV metrics (autonomic activity)
- Movement features (activity detection)
- Correlation features (multi-sensor fusion)
- Statistical features (baseline)

**Output:** Feature matrix ready for ML models

---

### HRV-Based Sleep/Wake Detection

**Goal:** Simple sleep/wake classification using only HR

**Workflow:** [HRV Analysis](common-workflows.md#1-heart-rate-variability-hrv-analysis)

**Features Extracted:**
- HR mean (higher when awake)
- HR variability (higher in sleep)
- HR coefficient of variation

**Output:** HRV features for binary classification

---

### Movement-Based Restlessness Detection

**Goal:** Detect restless sleep periods

**Workflow:** [Movement Analysis](common-workflows.md#2-movement-and-activity-detection)

**Features Extracted:**
- Movement magnitude (intensity)
- Activity count (discrete movements)
- Stillness ratio (% time still)

**Output:** Movement profiles for restlessness scoring

---

## 🛠️ Tools and Tips

### VS Code Integration

Add this to `.vscode/settings.json` for YAML validation:

```json
{
  "yaml.schemas": {
    "workflow-schema.json": "workflows/*.yaml"
  }
}
```

### Jupyter Notebook Template

```python
# Cell 1: Setup
from sleep_analysis.core import SignalCollection
from sleep_analysis.importers import MergingImporter
from sleep_analysis.signals import SignalType, SensorType, SensorModel, BodyPosition

collection = SignalCollection(metadata={'subject_id': 'subject_001'})

# Cell 2: Import data
# ... (see Python API Guide)

# Cell 3: Visualize
# ... (see Python API Guide)

# Cell 4: Extract features
# ... (see Python API Guide)

# Cell 5: Export
# ... (see Python API Guide)
```

### Bash Script for Batch Processing

```bash
#!/bin/bash
# process_all_subjects.sh

for subject_dir in data/subject_*/; do
    subject=$(basename $subject_dir)
    echo "Processing $subject..."

    python -m sleep_analysis.cli.run_workflow \
        --workflow workflows/complete_sleep_analysis.yaml \
        --data-dir "$subject_dir" \
        --output-prefix "$subject"
done

echo "All subjects processed!"
```

---

## 🆘 Getting Help

### Documentation

1. **[Troubleshooting Guide](../troubleshooting.md)** - Common errors and fixes
2. **[Feature Extraction Examples](../feature-extraction-examples.md)** - Detailed examples
3. **[Data Preparation Guide](../data-preparation.md)** - Import data correctly

### Community

- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share workflows
- Examples: See `workflows/` for working examples

### Debug Workflow Issues

```bash
# Run with verbose logging
python -m sleep_analysis.cli.run_workflow \
  --workflow my_workflow.yaml \
  --data-dir data \
  --log-level DEBUG
```

---

## 📈 What's Next?

After completing the tutorials:

1. **Customize workflows** for your specific data and research questions
2. **Integrate with ML pipelines** using the Python API
3. **Contribute** your custom feature extractors or workflows
4. **Scale up** to multi-subject, multi-night analyses
5. **Deploy** your analysis pipeline in production

---

## 🤝 Contributing

Found an error in the tutorials? Have a suggestion?

1. Open an issue on GitHub
2. Submit a pull request
3. Share your custom workflows

---

## 📜 License

This documentation is part of the Adaptive Sleep Algorithms framework.

---

**Ready to start?** → Begin with **[Getting Started](getting-started.md)** 🚀

**Have questions?** → Check **[Troubleshooting](../troubleshooting.md)** ❓

**Need examples?** → See **[Common Workflows](common-workflows.md)** 📋
