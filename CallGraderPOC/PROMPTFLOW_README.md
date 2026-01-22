# PromptFlow QA Call Grader Setup Guide

This PromptFlow template analyzes call center transcripts using Azure OpenAI to provide QA auditing, including summaries, customer questions, quality scores, and negative flags.

## ⚠️ Configuration Required

Before running this flow, you **MUST** update the following placeholders with your actual values:

### 1. Azure OpenAI Connection (`flow.dag.yaml`)

**Location:** `flow.dag.yaml` - Line 11

```yaml
connection: <YOUR_AZURE_OPENAI_CONNECTION_NAME>
```

**What to replace:**
- Replace `<YOUR_AZURE_OPENAI_CONNECTION_NAME>` with your Azure OpenAI connection name
- Example: `my-company-aoai-connection`
- **How to find it:** Check your Azure ML workspace under "Compute" → "Connections" or your PromptFlow connections

---

### 2. Azure OpenAI Deployment Name (`flow.dag.yaml`)

**Location:** `flow.dag.yaml` - Line 8

```yaml
deployment_name: <YOUR_DEPLOYMENT_NAME>
```

**What to replace:**
- Replace `<YOUR_DEPLOYMENT_NAME>` with your Azure OpenAI deployment name
- Example: `gpt-4`, `gpt-35-turbo`, or your custom deployment name
- **How to find it:** Check your Azure OpenAI resource in the Azure portal under "Deployments"

---

### 3. Environment Stage (`flow.meta.yaml`)

**Location:** `flow.meta.yaml` - Line 7

```yaml
promptflow.stage: <YOUR_ENVIRONMENT_STAGE>
```

**What to replace:**
- Replace `<YOUR_ENVIRONMENT_STAGE>` with your environment designation
- Common values: `dev`, `staging`, `prod`, `test`
- Example: `prod`

---

### 4. Section Name (`flow.meta.yaml`)

**Location:** `flow.meta.yaml` - Line 8

```yaml
promptflow.section: <YOUR_SECTION_NAME>
```

**What to replace:**
- Replace `<YOUR_SECTION_NAME>` with your organizational section or team name
- Example: `qc-team`, `call-center-ops`, `qa-auditing`

---

## 📁 File Structure

```
.
├── flow.dag.yaml           # Main flow configuration (UPDATE REQUIRED)
├── flow.meta.yaml          # Flow metadata (UPDATE REQUIRED)
├── LLM.jinja2             # Azure OpenAI prompt template
├── echo.py                # Echo tool
├── joke.jinja2            # Example joke prompt template
├── samples.json           # Sample batch input data
├── requirements.txt       # Python dependencies
└── PROMPTFLOW_README.md   # This file
```

## 🚀 Quick Start

1. **Update Configuration Files:**
   - Edit `flow.dag.yaml` and replace the two placeholders
   - Edit `flow.meta.yaml` and replace the two placeholders

2. **Prepare Input Data:**
   - Add call transcripts to `samples.json` in the format:
   ```json
   [
     {
       "transcript_text": "Customer: Hello... Agent: Hi there..."
     }
   ]
   ```

3. **Run the Flow:**
   - Use PromptFlow UI or CLI to execute the flow
   - The flow will analyze each transcript and output JSON results

## 📊 Expected Output

The flow outputs a JSON object per transcript:

```json
{
  "summary": "Brief summary of the call",
  "questions": ["Question 1?", "Question 2?"],
  "score": 85,
  "flags": ["Angry Customer", "Escalation Requested"]
}
```

## 🔐 Security Best Practices

- ✅ Never commit actual connection names or deployment names to version control
- ✅ Use environment variables or Azure Key Vault for sensitive configuration
- ✅ Keep credentials in `.gitignore` files
- ✅ Use connection secrets for storing Azure OpenAI access keys

## 📝 Customization

### To modify the QA scoring criteria:
Edit [LLM.jinja2](LLM.jinja2) and update the system prompt

### To change input/output format:
Edit [flow.dag.yaml](flow.dag.yaml) in the `inputs` and `outputs` sections

### To add additional processing steps:
Add new nodes to the flow and reference them in the DAG

---

**Last Updated:** January 22, 2026
