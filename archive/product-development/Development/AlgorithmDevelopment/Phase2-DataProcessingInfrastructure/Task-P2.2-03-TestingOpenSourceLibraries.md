
In parallel to developing our own algorithm development infrastructure, our data processing pipeline should also enable the conversion of our data for testing existing open source libraries, tools and algorithms. These will mostly be python libraries and open-source repositories published in academic literature. 

## Libraries to Test

See [[OpenSourceAlgorithmRepositories]]


### **1. Asleep**
#### Assessment
Initial results show that the algorithm over-estimates light sleep from awake - detecting light sleep when I was just lying in bed. Another night detected no sleep at all, showing a lack of robustness. This algorithm shows the limitations of accel-only sleep staging. 
### 2.  Sundararajan
https://github.com/wadpac/Sundararajan-SleepClassification-2021
This random forests implementation is the next one to test. The published paper is here:
https://www.nature.com/articles/s41598-020-79217-x


### **3. pyActigraphy**

### **4. YASA (Yet Another Spindle Algorithm)**

### **5. SciKit Digital Health (SKDH)**



## Progress

1. Tested Asleep