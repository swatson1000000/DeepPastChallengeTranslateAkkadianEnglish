# CLAUDE.md - Execution Guidelines

## Project Development Approach

This project is being built using **Python scripts** (`.py` files), not Jupyter notebooks (`.ipynb` files). This approach provides:

- **Production-ready code** - Scripts are more maintainable and reproducible
- **Better version control** - Python files track changes more cleanly
- **Scalability** - Scripts can be easily integrated into pipelines
- **Background execution** - Scripts run with `nohup` for long-running tasks
- **Logging** - All output is captured to timestamped log files

Jupyter notebooks may be used later for exploratory analysis, but the core implementation uses standalone Python scripts.

## Environment Setup

### Conda Environment
This project uses the **phi4** conda environment. Before executing any Python scripts, activate this environment:

```bash
conda activate phi4
```

## Python Script Execution Policy

All Python scripts executed for this project **MUST** be run in the background using `nohup` with log files written to the project log directory. The `phi4` conda environment must be active.

### Log Directory
```
/home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish/log
```

### Standard Execution Format

#### Prerequisites:
```bash
conda activate phi4
cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish
```

#### For any Python script, use:
```bash
nohup python <script_name> [arguments] > /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish/log/<script_name>_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Or more concisely from the project directory (with phi4 active):
```bash
nohup python <script_name> [arguments] > log/<script_name>_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Examples

#### Setup and run preprocessing:
```bash
conda activate phi4
cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish
nohup python src/preprocess.py --input data/raw/train.csv --output data/processed/train_clean.csv > log/preprocess_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Setup and run ByT5 training:
```bash
conda activate phi4
cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish
nohup python -u src/train_byt5.py --epochs 20 --output-dir models/byt5-akkadian > log/train_byt5_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Setup and run inference:
```bash
conda activate phi4
cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish
nohup python -u src/inference.py --model models/byt5-akkadian/best > log/inference_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Why This Approach?

1. **nohup** - Ensures process continues even if terminal disconnects
2. **Background execution** - Frees terminal for other tasks
3. **Timestamped logs** - Each run creates unique log file with timestamp
4. **Centralized logging** - All logs in `/log` directory for easy tracking
5. **Both stdout & stderr** - `2>&1` captures all output

### Monitoring Execution

#### View logs in real-time:
```bash
tail -f log/<log_file_name>.log
```

#### Check background processes:
```bash
jobs -l
ps aux | grep python
```

#### Stop a running process:
```bash
kill <PID>
# or force kill if needed:
kill -9 <PID>
```

### Log Directory Structure

The log directory will contain timestamped files like:
```
log/
├── preprocessing_20260202_120000.log
├── training_seq2seq_20260202_130000.log
├── training_mbart_20260202_140000.log
├── inference_20260202_150000.log
└── ...
```

### Important Notes

- Always create log files with timestamps to avoid overwriting previous runs
- Check log files regularly for errors or unexpected behavior
- Keep log files for reference and debugging
- Clean up old logs periodically if disk space becomes an issue
- The log directory is already created in the project structure

---

## ⚠️ CRITICAL: Clean Log Directory Before Restarting Training

**Every time you restart training, ALWAYS clean up the old log files first.**

This prevents log file confusion and ensures you're tracking the correct training run.

### Clean Logs Before Training

Before executing any training scripts, run:

```bash
cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish
rm -f log/train_*.log
```

### Complete Workflow for Training Restart

```bash
# Step 1: Activate environment
conda activate phi4

# Step 2: Navigate to project
cd /home/swatson/work/MachineLearning/kaggle/DeepPastChallengeTranslateAkkadianEnglish

# Step 3: Clean old training logs
rm -f log/train_*.log

# Step 4: Start ByT5 training
nohup python -u src/train_byt5.py --epochs 20 --output-dir models/byt5-akkadian > log/train_byt5_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Step 5: Verify logs were created
tail -f log/train_byt5_*.log
```

### Why Clean Logs?

1. **Avoid confusion** - Old logs from previous runs won't interfere
2. **Accurate monitoring** - `tail -f log/train_seq2seq_*.log` shows current run only
3. **Cleaner tracking** - Each training session has fresh logs
4. **Prevent misinterpretation** - No mixing of loss curves or metrics from old runs
5. **Easier debugging** - If training fails, you know which log to check

### Quick Commands

```bash
# Clean all training logs
rm -f log/train_*.log

# Clean all logs (more aggressive)
rm -f log/*.log

# View cleaned log directory
ls -la log/
```

---

**Effective Date**: February 2, 2026  
**Status**: Active  
**Last Updated**: February 12, 2026
