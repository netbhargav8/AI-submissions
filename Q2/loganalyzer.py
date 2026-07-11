# %% [markdown]
# #### **Importing the packages**
# %%
import os
import pandas as pd
# %% [markdown]
# 
# #### **1. Mocking a Sample Log File for Testing**
# %%
#This safely generates a sample log file if you don't have one in your folder yet
if not os.path.exists('app_log.txt'):
    print("No 'app_log.txt' detected. Generating a standard sample log file for you...")
    sample_logs = [
        "2026-07-11 08:12:00 [INFO] [AuthModule] User log_in attempt from IP 192.168.1.50\n",
        "2026-07-11 08:12:05 [WARNING] [AuthModule] Invalid password try for user admin\n",
        "2026-07-11 08:13:22 [ERROR] [DatabaseModule] Connection timeout encountered during pool initialization\n",
        "2026-07-11 08:14:10 [INFO] [PaymentModule] Processing gateway handshake\n",
        "2026-07-11 08:14:15 [ERROR] [PaymentModule] Gateway rejected transaction token ref_98412\n",
        "2026-07-11 08:15:01 [ERROR] [DatabaseModule] Deadlock detected during concurrent batch update\n",
        "2026-07-11 08:16:44 [WARNING] [NetworkModule] High latency spike observed on external API endpoint\n",
        "2026-07-11 08:18:12 [ERROR] [PaymentModule] Gateway communication failure - Timeout 504\n"
    ]
    with open('app_log.txt', 'w') as file:
        file.writelines(sample_logs)
# %% [markdown]
# 
# #### **2. Reading and Parsing the Log File**
# 
# %%
print("Executing Step 1: Parsing application log files...")
parsed_logs = []

# Open and scan the file line by line
with open('app_log.txt', 'r') as file:
    for line in file:
        # Check for our targeted error metrics
        if '[ERROR]' in line or '[WARNING]' in line:
            # Determine Log Level
            log_level = 'ERROR' if '[ERROR]' in line else 'WARNING'

            # Simple text parsing strategy to extract the bracketed module name
            # Example: from "... [AuthModule] ...", extract "AuthModule"
            try:
                start_idx = line.find('[', line.find(log_level)) + 1
                end_idx = line.find(']', start_idx)
                module_name = line[start_idx:end_idx]
            except Exception:
                module_name = 'UnknownModule'

            # Append structured record
            parsed_logs.append({
                'Log Level': log_level,
                'Module': module_name
            })

# Convert structured data into a Pandas DataFrame
dataset = pd.DataFrame(parsed_logs)
# %% [markdown]
# 
# #### **3. Frequency Data Aggregation**
# 
# %%
print("Executing Step 2: Aggregating diagnostic error metrics...")

if not dataset.empty:
    # Group by Module and Log Level, then count matching occurrences
    frequency_report = dataset.groupby(['Module', 'Log Level']).size().reset_index(name='Frequency')

    # Sort by the highest error frequency to bring problem areas to the top
    frequency_report = frequency_report.sort_values(by='Frequency', ascending=False)

    # Print console readout matrix
    print("\n" + "="*50)
    print("      DIAGNOSTIC LOG ERROR & WARNING FREQUENCY     ")
    print("="*50)
    print(frequency_report.to_string(index=False))
    print("="*50 + "\n")

    # ==========================================
    # 4. Generating the CSV Report
    # ==========================================
    output_filename = 'log_error_frequency.csv'
    frequency_report.to_csv(output_filename, index=False)
    print(f"Pipeline Success: Diagnostic CSV metrics written to '{output_filename}'.")
else:
    print("\n[Alert] No ERROR or WARNING logs were detected inside 'app_log.txt'.\n")
# %%
