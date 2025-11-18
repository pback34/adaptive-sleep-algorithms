
Data will flow from raw signals to engineered features, with often many stages of conversion in between. We need to define a hierarchy of signals.



| Level 1 - Raw Signals | Level 2 - Derived Signals | Level 3        | Level 4                                                                      | Epoch Features         |
| --------------------- | ------------------------- | -------------- | ---------------------------------------------------------------------------- | ---------------------- |
| PPG/EKG               | Pulse Intervals           | HR             | HRV                                                                          | Mean, Median, Min Max  |
| PPG/EKG               | Pulse Intevals            | HR             | HR                                                                           | Mean, Median, Min, Max |
| PPG/EKG               | Pulse Intervals           | HR             | % of baseline                                                                | Mean, Median, Min, Max |
| PPG/EKG               | Pulse Intevals            | HR             | Diff of previous time window (compare last minute to the minute before that) | Mean, Median, Min, Max |
| Accel XYZ             | Magnitude                 | Trshold Counts |                                                                              |                        |
| Accel XYZ             | Position                  |                |                                                                              |                        |
| Accel XYZ             | Velocity                  |                |                                                                              |                        |
| Accel XYZ             |                           |                |                                                                              |                        |
