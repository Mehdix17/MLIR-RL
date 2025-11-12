# GitHub Push Checklist

## ⚠️ ISSUES FOUND

### 🔴 Critical Issues (MUST FIX)

1. **Hardcoded Absolute Paths**
   - ❌ `scripts/train.sh` - Line 27, 32, 33
   - ❌ `scripts/eval.sh` - Line 28, 33
   - ❌ `scripts/neptune-sync.sh` - Line 24
   - ❌ `rl_autoschedular/state.py` - Line 219 (error message)
   - ❌ `docs/MLIR_Python_Setup_Steps.md` - Lines 21, 28

2. **Secrets Exposed in .env**
   - ❌ `.env` contains Neptune API token (will be ignored by .gitignore but be careful)
   - ❌ Email address in SLURM scripts: `mb10856@nyu.edu`

3. **Very Large Directories (11.6GB total!)**
   - ⚠️ `llvm-project/` = 8.6GB (should NOT be pushed)
   - ⚠️ `data/` = 3.0GB (most should NOT be pushed)
   - ⚠️ `results/` = 842MB (training outputs - don't push)

### 🟡 Recommended Fixes

4. **Missing Documentation**
   - ⚠️ No LICENSE file
   - ⚠️ No CONTRIBUTING.md
   - ⚠️ README could be more comprehensive (created README_NEW.md)

5. **Build Artifacts**
   - ⚠️ Model checkpoints (*.pt files) in results/
   - ⚠️ Log files (*.out, *.err)

## ✅ WHAT'S ALREADY GOOD

- ✅ `.gitignore` exists (but needs improvement - DONE)
- ✅ Project structure is well organized
- ✅ Documentation exists in `docs/`
- ✅ Python package structure is clean
- ✅ Requirements.txt is present

## 🔧 FIXES APPLIED

### 1. Improved .gitignore ✅
```
Added:
- llvm-project/build/ (8.6GB!)
- results/, logs/ (training artifacts)
- data/all/, data/polybench/ (large datasets)
- *.pt, *.pth (model checkpoints)
- .neptune/ (tracking cache)
```

### 2. Created .env.example ✅
- Template without secrets
- Safe to commit

### 3. Created prepare_for_github.sh ✅
- Converts absolute paths to relative
- Updates all scripts automatically

### 4. Created README_NEW.md ✅
- Comprehensive project overview
- Setup instructions
- Architecture explanation
- Results showcase

## 📋 STEP-BY-STEP: Preparing for GitHub

### Step 1: Run Preparation Script

```bash
cd /scratch/mb10856/MLIR-RL
bash prepare_for_github.sh
```

This will:
- Make all paths relative in scripts
- Create .env.example
- Update documentation paths

### Step 2: Manual Fixes

#### Update Email in SLURM Scripts
```bash
# Edit these files and replace with generic or remove:
# scripts/train.sh - Line 15
# scripts/eval.sh - Line 15  
# scripts/neptune-sync.sh - Line 11

# Option 1: Use a generic email
sed -i 's/mb10856@nyu.edu/your-email@example.com/g' scripts/*.sh

# Option 2: Remove email notification (comment out)
sed -i 's/^#SBATCH --mail-user=/#SBATCH #--mail-user=/g' scripts/*.sh
```

#### Fix Documentation Path
```bash
# Edit docs/MLIR_Python_Setup_Steps.md
# Replace /scratch/mb10856/MLIR-RL with ./
```

### Step 3: Replace README
```bash
mv README.md README_OLD.md
mv README_NEW.md README.md
```

### Step 4: Verify .gitignore

```bash
# Check what would be committed (should be small!)
git status

# Check size (should be < 100MB)
git ls-files | xargs du -ch | tail -1
```

**Expected to be tracked:**
- ✅ Source code (*.py)
- ✅ Configuration files
- ✅ Documentation
- ✅ Scripts (*.sh)
- ✅ Small test dataset (data/test/)
- ✅ Requirements

**Should NOT be tracked:**
- ❌ llvm-project/build/ (8.6GB)
- ❌ results/ (842MB)
- ❌ data/all/ (3GB)
- ❌ logs/
- ❌ .env
- ❌ *.pt files

### Step 5: Create GitHub Repository

```bash
# On GitHub, create a new repository: MLIR-RL
# Then:

cd /scratch/mb10856/MLIR-RL

# Initialize if not already a git repo
git init

# Add remote
git remote add origin https://github.com/YOUR-USERNAME/MLIR-RL.git

# Check what will be added
git status
git add -n .  # Dry run

# Add files (will respect .gitignore)
git add .

# Commit
git commit -m "Initial commit: MLIR reinforcement learning optimizer"

# Push
git push -u origin main
```

### Step 6: Post-Push Setup

Add to GitHub repository:

1. **Add LICENSE file** on GitHub
   - Settings → Add license
   - Recommend: MIT or Apache 2.0

2. **Add repository description**
   ```
   Reinforcement learning for MLIR compiler optimization using PPO
   ```

3. **Add topics/tags**
   - mlir
   - compiler-optimization
   - reinforcement-learning
   - ppo
   - llvm
   - machine-learning

4. **Create .github/workflows/** (optional)
   - Add CI/CD if needed

## 🎯 FINAL CHECKLIST

Before pushing, verify:

- [ ] Ran `prepare_for_github.sh`
- [ ] Replaced email addresses in SLURM scripts
- [ ] Fixed paths in docs/MLIR_Python_Setup_Steps.md
- [ ] Moved README_NEW.md to README.md
- [ ] Verified .env is in .gitignore
- [ ] Checked `git status` (no large files)
- [ ] Tested that scripts still work with relative paths
- [ ] Added LICENSE file
- [ ] Updated README with your contact info

## 📊 Repository Size Estimate

**After .gitignore (safe to push):**
- Source code: ~5MB
- Documentation: ~1MB
- Scripts: <1MB
- Test data: ~50MB
- Tools source: ~2MB

**Total: ~60MB** ✅ (GitHub limit is 1GB per repo)

## ⚠️ IMPORTANT NOTES

1. **Never commit .env** - Contains API tokens!
2. **LLVM build is huge** - Let users build their own
3. **Training results** - Share via Neptune.ai, not Git
4. **Large models** - Use Git LFS or external storage

## 🔒 Security Check

Before pushing, verify no secrets:

```bash
# Check for potential secrets
grep -r "NEPTUNE_TOKEN\|api_token\|password" --include="*.py" --include="*.sh" --include="*.json" .

# Should only find references, not actual tokens
```

## 📞 Need Help?

If you encounter issues:
1. Check [GitHub's guide](https://docs.github.com/en/get-started)
2. Use `git status` to see what's being tracked
3. Use `git rm --cached <file>` to untrack files
4. Run `git add -n .` to do a dry run first

---

**Ready to push?** Follow the steps above in order! 🚀
