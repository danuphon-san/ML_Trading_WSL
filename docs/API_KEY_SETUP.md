# API Key Setup Guide

## Overview

This project uses environment variables to securely manage API keys for data providers. **Never commit API keys directly to git!**

---

## 🔐 **Setup Instructions**

### Step 1: Create `.env` File

Copy the example file and add your actual API keys:

```bash
cd /home/user/ML_Trading_WSL
cp .env.example .env
```

### Step 2: Edit `.env` File

Open `.env` in your text editor and replace placeholder values:

```bash
nano .env
# or
code .env
```

Update with your actual keys:

```bash
# SimFin API Key (required if using SimFin)
SIMFIN_API_KEY=your-actual-simfin-key-here

# Alpha Vantage API Key (required if using Alpha Vantage)
ALPHA_VANTAGE_API_KEY=your-actual-alpha-vantage-key-here
```

### Step 3: Verify Setup

Test that the configuration loads correctly:

```bash
python -c "from utils.config_loader import load_config_with_validation; \
           cfg = load_config_with_validation(provider='simfin'); \
           print('✓ Config loaded successfully')"
```

If successful, you should see:
```
✓ Config loaded successfully
```

---

## 🔑 **Getting API Keys**

### SimFin API Key

1. Go to [https://simfin.com/](https://simfin.com/)
2. Sign up for an account
3. Subscribe to Start Plan ($14.99/month) or higher
4. Go to your dashboard and copy your API key
5. Paste into `.env` file

### Alpha Vantage API Key (Optional)

1. Go to [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
2. Enter your email and get a free API key
3. Paste into `.env` file

---

## ⚙️ **How It Works**

### Configuration Loading

The system uses `utils/config_loader.py` to:

1. **Load `.env` file** - Reads environment variables from `.env`
2. **Interpolate variables** - Replaces `${VAR_NAME}` in `config.yaml` with actual values
3. **Validate keys** - Ensures required keys are set and not placeholders

### Example

**config/config.yaml:**
```yaml
fundamentals:
  simfin:
    api_key: ${SIMFIN_API_KEY}  # Placeholder
```

**.env:**
```bash
SIMFIN_API_KEY=***REMOVED***
```

**Result:**
```python
cfg['fundamentals']['simfin']['api_key']  # Returns: "6174fd1a-e6a9..."
```

---

## 🚨 **Security Best Practices**

### ✅ DO:
- ✅ Store API keys in `.env` file
- ✅ Add `.env` to `.gitignore` (already done)
- ✅ Use different keys for development and production
- ✅ Rotate keys periodically
- ✅ Share `.env.example` as a template

### ❌ DON'T:
- ❌ Commit `.env` to git
- ❌ Hard-code API keys in source code
- ❌ Share API keys in chat/email
- ❌ Use production keys in public repos
- ❌ Store keys in plain text notes

---

## 🔄 **Remove API Keys from Git History** (IMPORTANT!)

If you accidentally committed API keys to git, follow these steps to remove them from history:

### Option 1: Using git-filter-repo (Recommended)

```bash
# Install git-filter-repo
pip install git-filter-repo

# Remove sensitive file from history
git filter-repo --path config/config.yaml --invert-paths --force

# Or remove specific pattern (API key text)
git filter-repo --replace-text <(echo "***REMOVED***==>REMOVED")
```

### Option 2: Using BFG Repo-Cleaner

```bash
# Download BFG
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar

# Remove passwords/keys
echo "***REMOVED***" > passwords.txt
java -jar bfg-1.14.0.jar --replace-text passwords.txt .git

# Clean up
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

### Option 3: Rewrite Entire History (Nuclear Option)

⚠️ **Warning:** This rewrites all commit hashes. Only use if absolutely necessary.

```bash
# Create a backup first!
git clone --mirror . ../ML_Trading_WSL_backup.git

# Remove file from all commits
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch config/config.yaml" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (WARNING: Destructive!)
git push origin --force --all
git push origin --force --tags
```

### After Removal:

1. **Invalidate the old keys** - Rotate them on provider websites
2. **Update `.env`** with new keys
3. **Verify cleanup:**
   ```bash
   git log --all --full-history -- config/config.yaml
   # Should not show any commits with API keys
   ```

---

## 🐳 **Docker / Container Usage**

### Pass environment variables to container:

```bash
# Option 1: Using .env file
docker run --env-file .env your-trading-app

# Option 2: Individual variables
docker run -e SIMFIN_API_KEY=$SIMFIN_API_KEY your-trading-app

# Option 3: Docker Compose
# In docker-compose.yml:
services:
  trading-app:
    env_file:
      - .env
```

---

## 🖥️ **Server / Production Setup**

### Option 1: System Environment Variables

```bash
# Add to ~/.bashrc or ~/.zshrc
export SIMFIN_API_KEY="your-key-here"
export ALPHA_VANTAGE_API_KEY="your-key-here"

# Reload shell
source ~/.bashrc
```

### Option 2: Systemd Service (Linux)

```ini
# /etc/systemd/system/trading-app.service
[Service]
Environment="SIMFIN_API_KEY=your-key-here"
Environment="ALPHA_VANTAGE_API_KEY=your-key-here"
ExecStart=/path/to/python scripts/daily_update_data.py
```

### Option 3: AWS Secrets Manager / Azure Key Vault

For enterprise deployments, use cloud secret management:

```python
# Example with AWS Secrets Manager
import boto3

client = boto3.client('secretsmanager')
response = client.get_secret_value(SecretId='trading-app/simfin-key')
simfin_key = response['SecretString']
```

---

## 🧪 **Testing**

### Test Configuration Loading

```bash
# Test SimFin config
python -c "from utils.config_loader import load_config_with_validation; \
           cfg = load_config_with_validation(provider='simfin'); \
           print(f'SimFin Key: {cfg[\"fundamentals\"][\"simfin\"][\"api_key\"][:10]}...')"

# Test Alpha Vantage config
python -c "from utils.config_loader import load_config_with_validation; \
           cfg = load_config_with_validation(provider='alpha_vantage'); \
           print(f'AV Key: {cfg[\"fundamentals\"][\"alpha_vantage\"][\"api_key\"][:10]}...')"
```

### Test Data Ingestion

```bash
# Test SimFin ingestion with real API key
python scripts/ingest_fundamentals_simfin.py --symbols AAPL
```

---

## 🆘 **Troubleshooting**

### Error: "Environment variable 'SIMFIN_API_KEY' not found"

**Solution:**
1. Check that `.env` file exists: `ls -la .env`
2. Verify content: `cat .env | grep SIMFIN_API_KEY`
3. Ensure no extra spaces: `SIMFIN_API_KEY=key` (not `SIMFIN_API_KEY = key`)

### Error: "Unresolved environment variable in fundamentals.simfin.api_key"

**Solution:**
1. Check `.env` syntax
2. Try setting directly in shell:
   ```bash
   export SIMFIN_API_KEY="your-key"
   python scripts/ingest_fundamentals_simfin.py
   ```

### Error: "Configuration validation failed: Placeholder value"

**Solution:**
Replace placeholder text in `.env`:
```bash
# Wrong:
SIMFIN_API_KEY=your-simfin-api-key-here

# Correct:
SIMFIN_API_KEY=***REMOVED***
```

---

## 📚 **Additional Resources**

- [12-Factor App: Config](https://12factor.net/config)
- [OWASP: Secrets Management](https://owasp.org/www-community/vulnerabilities/Use_of_hard-coded_password)
- [GitHub: Removing Sensitive Data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)

---

## ✅ **Checklist**

Before pushing to git:

- [ ] `.env` file created and populated with real keys
- [ ] `.env` is in `.gitignore`
- [ ] `config.yaml` uses `${VAR_NAME}` syntax (no hard-coded keys)
- [ ] Tested config loading: `python -c "from utils.config_loader import load_config; cfg = load_config()"`
- [ ] Verified `.env` not in git: `git status` (should not show .env)
- [ ] Old API keys removed from git history (if applicable)
- [ ] Old API keys invalidated on provider websites

---

**Need help?** Check `docs/SIMFIN_INTEGRATION.md` for SimFin-specific setup or open an issue.
