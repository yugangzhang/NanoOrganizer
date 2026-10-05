# Advanced File Filters Guide

## 🎯 Overview

The folder browser now supports advanced filename filtering beyond just file extensions. You can filter by:
1. **Extension** (*.csv, *.npz, etc.)
2. **String patterns** in filenames (contains, not contains, AND/OR logic)

---

## 🔍 Filter Types

### 1. Extension Filter (Basic)
Select from dropdown: `*.csv`, `*.npz`, `*.txt`, `*.dat`, `*.*`

**Example**:
- `*.csv` → Only CSV files

---

### 2. Advanced Filters (Click "🔍 Advanced Filters" expander)

#### **Must contain ALL** (AND logic)
Filename must contain **ALL** specified strings (comma-separated).

**Input**: `sample, 2024, run1`
**Matches**:
- ✅ `sample_2024_run1.csv`
- ✅ `run1_sample_2024.csv`
- ❌ `sample_2024.csv` (missing 'run1')
- ❌ `2024_run1.csv` (missing 'sample')

---

#### **Must contain ANY** (OR logic)
Filename must contain **AT LEAST ONE** specified string.

**Input**: `gold, silver, copper`
**Matches**:
- ✅ `gold_nanoparticle.csv`
- ✅ `silver_sample.csv`
- ✅ `gold_silver_alloy.csv`
- ❌ `aluminum.csv` (has none)

---

#### **Must NOT contain** (Exclusion)
Filename must **NOT** contain any specified strings.

**Input**: `temp, backup, old`
**Matches**:
- ✅ `sample_final.csv`
- ❌ `sample_temp.csv` (has 'temp')
- ❌ `backup_data.csv` (has 'backup')
- ❌ `old_sample.csv` (has 'old')

---

## 💡 Common Use Cases

### Use Case 1: Select specific sample series
**Goal**: Load all "Au" (gold) samples from 2024

**Settings**:
- Extension: `*.csv`
- Must contain ALL: `Au, 2024`

**Result**:
- ✅ `Au_sample_2024_01.csv`
- ✅ `2024_Au_run_final.csv`
- ❌ `Ag_2024.csv` (wrong material)
- ❌ `Au_2023.csv` (wrong year)

---

### Use Case 2: Compare different conditions
**Goal**: Load samples with either "high_temp" or "low_temp"

**Settings**:
- Extension: `*.csv`
- Must contain ANY: `high_temp, low_temp`

**Result**:
- ✅ `sample_high_temp_01.csv`
- ✅ `sample_low_temp_01.csv`
- ❌ `sample_room_temp.csv`

---

### Use Case 3: Exclude test/temporary files
**Goal**: Load real data, exclude temporary/test files

**Settings**:
- Extension: `*.csv`
- Must NOT contain: `temp, test, backup, old`

**Result**:
- ✅ `sample_001.csv`
- ✅ `final_data.csv`
- ❌ `temp_analysis.csv`
- ❌ `test_run.csv`
- ❌ `backup_sample.csv`

---

### Use Case 4: Complex filtering
**Goal**: Load gold samples from run 1-5, but not backups

**Settings**:
- Extension: `*.csv`
- Must contain ALL: `Au`
- Must contain ANY: `run1, run2, run3, run4, run5`
- Must NOT contain: `backup, temp`

**Result**:
- ✅ `Au_sample_run1.csv`
- ✅ `Au_run3_final.csv`
- ❌ `Au_run6.csv` (wrong run number)
- ❌ `Au_run1_backup.csv` (is backup)

---

## 🎨 How to Use

1. **Open CSV Plotter** (or any tool with folder browser)
2. Select **"Browse server"**
3. Choose **extension filter** from dropdown (e.g., `*.csv`)
4. Click **"🔍 Advanced Filters"** expander
5. Enter your patterns:
   - Comma-separated: `pattern1, pattern2, pattern3`
   - No quotes needed
   - Case-sensitive
6. See **live filtering** as you type
7. Select files with **checkboxes**
8. Click **"📥 Load Selected Files"**

---

## 🔄 Reset Filters

Click **"🔄 Reset Filters"** button in the Advanced Filters section to clear all pattern filters.

---

## ⚡ Tips

1. **Start broad, then narrow**:
   - First: Select extension (*.csv)
   - Then: Add "contains" filters
   - Finally: Add "not contains" to exclude

2. **Combine filters**:
   - Use all three types together for precise selection
   - Extension + Contains ALL + NOT contains = very specific

3. **Test incrementally**:
   - Add one filter at a time
   - Check the filtered count
   - Adjust as needed

4. **Common patterns**:
   - Date ranges: `2024, Jan` or `2024_01`
   - Sample types: `Au, Ag, Cu`
   - Exclude: `temp, backup, test, old, copy`

---

## 📊 Filter Logic Summary

```
Final Files = Extension Filter
              AND (Must contain ALL patterns)
              AND (Must contain ANY pattern OR no ANY patterns specified)
              AND NOT (Must NOT contain any pattern)
```

**Example**:
```
Extension: *.csv
Must contain ALL: sample, 2024
Must contain ANY: gold, silver
Must NOT contain: temp

→ Files that are:
  - CSV files
  - AND have both "sample" and "2024" in name
  - AND have either "gold" or "silver" (or both)
  - AND do NOT have "temp"
```

---

## 🎉 Benefits

- ✅ **Precise selection** - Get exactly the files you need
- ✅ **No manual sorting** - Filter automatically
- ✅ **Exclude unwanted** - Remove temp/backup files easily
- ✅ **Fast workflow** - Select multiple files with one filter
- ✅ **Reusable patterns** - Same filters work across all tools

---

**Enjoy your advanced filtering! 🚀**
