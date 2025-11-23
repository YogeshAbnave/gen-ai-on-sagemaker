# Design Document: Production-Grade Model Fine-Tuning on AWS

## Overview

This design document provides comprehensive AWS GUI step-by-step instructions for deploying a production-grade Model Fine-Tuning and Customization system using Amazon SageMaker AI and Amazon Bedrock. The design implements complete MLOps practices including data preparation, distributed training, model evaluation, deployment, monitoring, and continuous improvement.

The Fine-Tuning system architecture consists of:
- **Data Layer**: S3 for training data, validation data, and model artifacts
- **Training Layer**: SageMaker Training for custom training, Bedrock for managed fine-tuning
- **Evaluation Layer**: Automated model evaluation with custom metrics
- **Registry Layer**: SageMaker Model Registry for version control and lineage
- **Deployment Layer**: SageMaker Endpoints and Bedrock Provisioned Throughput
- **Monitoring Layer**: CloudWatch, SageMaker Model Monitor for drift detection
- **Orchestration Layer**: SageMaker Pipelines for FMOps automation
- **Security Layer**: IAM, KMS encryption, VPC isolation

## Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          AWS Cloud (Region)                          │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    VPC (10.0.0.0/16)                           │ │
│  │                                                                 │ │
│  │  ┌──────────────────┐         ┌──────────────────┐            │ │
│  │  │  Private Subnet  │         │  Private Subnet  │            │ │
│  │  │  (10.0.10.0/24)  │         │  (10.0.11.0/24)  │            │ │
│  │  │                  │         │                  │            │ │
│  │  │  ┌────────────┐  │         │  ┌────────────┐ │            │ │
│  │  │  │ SageMaker  │  │         │  │ SageMaker  │ │            │ │
│  │  │  │  Training  │  │         │  │  Endpoint  │ │            │ │
│  │  │  └────────────┘  │         │  └────────────┘ │            │ │
│  │  └──────────────────┘         └──────────────────┘            │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  S3 Buckets  │  │   Bedrock    │  │  SageMaker   │              │
│  │ (Train/Model)│  │ (Fine-Tuning)│  │   Pipelines  │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Model Registry│  │  CloudWatch  │  │ Model Monitor│              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

### Fine-Tuning Workflow

```
Training Data → Data Validation → Data Preparation
                                         ↓
                              [Choose Training Method]
                                         ↓
                    ┌────────────────────┴────────────────────┐
                    ↓                                          ↓
            [Bedrock Fine-Tuning]                  [SageMaker Training]
                    ↓                                          ↓
            [Bedrock Custom Model]                  [Model Artifacts → S3]
                    ↓                                          ↓
                    └────────────────────┬────────────────────┘
                                         ↓
                              [Model Evaluation]
                                         ↓
                              [Register in Model Registry]
                                         ↓
                              [Deploy to Endpoint]
                                         ↓
                              [Monitor Performance]
                                         ↓
                              [Continuous Improvement]
```

## Components and Interfaces

### Phase 1: Foundation Infrastructure Setup

#### Component 1.1: VPC and Network Configuration

**AWS GUI Steps** (Abbreviated for efficiency):

1. Navigate to VPC → Create VPC
2. **Name**: "ml-training-vpc", **CIDR**: "10.0.0.0/16"
3. Create 2 private subnets in different AZs
4. Create NAT Gateway for outbound internet access
5. Configure route tables

#### Component 1.2: S3 Buckets for ML Workflow

**AWS GUI Steps**:

1. Navigate to S3 → Create bucket
2. Create buckets:
   - "ml-training-data-[account]-[region]"
   - "ml-model-artifacts-[account]-[region]"
   - "ml-evaluation-results-[account]-[region]"
3. Enable versioning and encryption on all buckets
4. Create folder structure:
   - training-data/raw/
   - training-data/processed/
   - models/checkpoints/
   - models/final/
   - evaluation/reports/

#### Component 1.3: IAM Roles for ML Operations

**AWS GUI Steps**:

1. Navigate to IAM → Roles → Create role
2. Create "SageMakerTrainingRole":
   - Trusted entity: SageMaker
   - Attach policies:
     - AmazonSageMakerFullAccess
     - AmazonS3FullAccess
     - AmazonBedrockFullAccess
3. Create "SageMakerDeploymentRole" with similar permissions
4. Add inline policy for KMS encryption access

### Phase 2: Data Preparation for Fine-Tuning

#### Component 2.1: Prepare Training Data

**Purpose**: Format and validate data for fine-tuning.

**AWS GUI Steps**:

1. **In SageMaker Studio**, create new notebook
   - Go to SageMaker Console → Click "Studio" → "Open Studio"
   - In Studio, click "File" → "New" → "Notebook"
   - Select kernel: Python 3 (Data Science) or PyTorch

2. **Install dependencies**:
```python
!pip install datasets pandas jsonlines boto3
```

3. **Verify Installation** (Run in notebook cell):
```python
import datasets
import pandas as pd
import jsonlines
import boto3

print("✓ datasets version:", datasets.__version__)
print("✓ pandas version:", pd.__version__)
print("✓ boto3 version:", boto3.__version__)
print("✓ All frameworks installed successfully!")
```

4. **Load and format data**:
```python
import json
import jsonlines
import pandas as pd

# Example: Prepare data for instruction fine-tuning
def format_for_finetuning(data):
    """
    Format data for Bedrock/SageMaker fine-tuning
    Required format: {"prompt": "...", "completion": "..."}
    """
    formatted_data = []
    
    for item in data:
        formatted_item = {
            "prompt": f"Instruction: {item['instruction']}\nInput: {item['input']}\n\nResponse:",
            "completion": f" {item['output']}"
        }
        formatted_data.append(formatted_item)
    
    return formatted_data

# Load your data
df = pd.read_csv('your_data.csv')

# Format data
formatted_data = format_for_finetuning(df.to_dict('records'))

# Save as JSONL for Bedrock
with jsonlines.open('training_data.jsonl', 'w') as writer:
    writer.write_all(formatted_data)

# Split into train/validation
train_size = int(0.9 * len(formatted_data))
train_data = formatted_data[:train_size]
val_data = formatted_data[train_size:]

with jsonlines.open('train.jsonl', 'w') as writer:
    writer.write_all(train_data)

with jsonlines.open('validation.jsonl', 'w') as writer:
    writer.write_all(val_data)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
```

5. **Upload to S3**:
```python
import boto3

s3 = boto3.client('s3')
bucket = 'ml-training-data-[account]-[region]'

s3.upload_file('train.jsonl', bucket, 'training-data/processed/train.jsonl')
s3.upload_file('validation.jsonl', bucket, 'training-data/processed/validation.jsonl')
```

### Phase 3: Fine-Tuning with Bedrock (Managed Approach)

#### Component 3.1: Create Bedrock Custom Model

**Purpose**: Fine-tune foundation models using Bedrock's managed service.

**AWS GUI Steps** (Updated for 2024 Console):

1. **Navigate to Bedrock Console**
   - Go to AWS Console → Search for "Bedrock"
   - Click "Amazon Bedrock"

2. **Request Model Access** (First-time setup)
   - In left sidebar, click "Model access" under "Foundation models"
   - Click "Manage model access" or "Edit"
   - Enable access for models you want to fine-tune:
     - Claude 3 Haiku (recommended for fine-tuning)
     - Titan Text G1 - Express
     - Titan Text models
   - Click "Save changes"
   - Wait for approval (usually instant, some models may take 24-48 hours)

3. **Access Custom Models** (Updated Path)
   - In left sidebar, look for "Foundation models" section
   - Click "Custom models" (may be under "Customization" section)
   - Click "Customize model" button (previously "Create custom model")

4. **Choose Customization Method**
   - Select **"Fine-tuning"** (recommended for most use cases)
   - Alternative: "Continued pre-training" (for domain adaptation)
   - Click "Next"

5. **Select Base Model**
   - Choose from models with "Customization available" badge:
     - **Claude 3 Haiku** (best for instruction following)
     - **Titan Text G1 - Express** (cost-effective)
     - **Titan Text Premier** (higher quality)
   - **Model name**: "custom-domain-model-v1"
   - Click "Next"

6. **Configure Training Data** (Updated Interface)
   - **Job name**: "finetuning-job-v1-[timestamp]"
   - **Training dataset**: 
     - Click "Browse S3"
     - Navigate to: s3://ml-training-data-[account]-[region]/training-data/processed/train.jsonl
     - Select file
   - **Validation dataset** (optional but recommended):
     - Click "Browse S3"
     - Select: s3://ml-training-data-.../validation.jsonl
   - **Output location**: 
     - Browse S3 → s3://ml-model-artifacts-[account]-[region]/bedrock-models/
   - **Service role**: 
     - Select existing role with Bedrock and S3 permissions
     - Or click "Create and use a new service role"
   - Click "Next"

7. **Configure Hyperparameters** (Simplified UI)
   - **Epochs**: 1-10 (default: 3, recommended: 3-5)
   - **Batch size**: Auto-configured based on model (typically 8-32)
   - **Learning rate multiplier**: 0.1-2.0 (default: 1.0)
     - Use 0.5-1.0 for small datasets
     - Use 1.0-1.5 for large datasets
   - **Early stopping**: Enable to prevent overfitting (recommended)
   - **Early stopping patience**: 2-3 epochs
   - Click "Next"

8. **Add Tags** (Optional)
   - Add tags for cost tracking and organization:
     - Key: "Project", Value: "ModelFineTuning"
     - Key: "Environment", Value: "Development"
   - Click "Next"

9. **Review and Submit**
   - Review all configurations carefully
   - Verify S3 paths are correct
   - Check hyperparameters
   - Click "Create customization job"
   - Job will appear in "Customization jobs" tab

10. **Monitor Training Progress** (Updated Monitoring)
    - Go to "Customization jobs" tab
    - Click on your job name
    - View status: InProgress → Completed/Failed
    - **Job details** section shows:
      - Start time and duration
      - Training data statistics
      - Hyperparameters used
    - **Training metrics** tab (new feature):
      - View loss curves
      - Monitor validation metrics
      - Check for overfitting
    - **CloudWatch logs** link for detailed logs
    - Training typically takes 2-6 hours depending on:
      - Dataset size (100-10,000+ examples)
      - Model size
      - Number of epochs

11. **Create Provisioned Throughput** (Changed Process)
    - Once job status shows "Completed"
    - Go to "Custom models" tab
    - Find your completed model
    - Click model name → Click "Create provisioned throughput" button
    - Or click "Actions" dropdown → "Create provisioned throughput"
    
    **Provisioned Throughput Configuration**:
    - **Provisioned throughput name**: "custom-model-endpoint-v1"
    - **Model units**: 
      - Start with 1 (minimum, ~2-5 requests/sec)
      - Scale up based on load: 2-10 units for production
    - **Commitment term**:
      - "No commitment" (hourly billing, flexible)
      - "1 month" (discounted rate)
      - "6 months" (maximum discount)
    - Click "Create"
    - Wait 10-20 minutes for provisioning
    - Status will change: Creating → InService

12. **Test Custom Model** (New Testing Interface)
    - Go to "Playgrounds" in left sidebar
    - Select "Chat" or "Text" playground
    - In model selector dropdown:
      - Click "Custom models" tab
      - Select your provisioned model from list
    - **Test with sample prompts**:
      - Enter domain-specific prompts
      - Compare responses with base model
      - Verify fine-tuning improvements
    - **Adjust inference parameters**:
      - Temperature: 0.0-1.0
      - Top P: 0.1-1.0
      - Max tokens: 100-4096
    - Save successful test cases for documentation

13. **API Integration** (For Application Use)
    ```python
    import boto3
    import json
    
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    # Your custom model ARN (found in Custom models details)
    model_id = "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/your-model-id"
    
    # For Claude models
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "messages": [{
            "role": "user",
            "content": "Your prompt here"
        }],
        "temperature": 0.7,
        "top_p": 0.9
    })
    
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=body
    )
    
    response_body = json.loads(response['body'].read())
    print(response_body['content'][0]['text'])
    ```

**Important Notes**:
- Data format must be JSONL with exact schema: `{"prompt": "...", "completion": "..."}`
- Minimum dataset size: 32 examples (recommended: 100-1,000+)
- Maximum file size: 10 GB
- Provisioned throughput has minimum 1-hour billing
- Custom models are region-specific
- Model artifacts are automatically stored in your S3 output location

### Phase 4: Fine-Tuning with SageMaker Training (Advanced Control)

#### Component 4.1: Create SageMaker Training Job

**Purpose**: Fine-tune models with full control using custom training scripts.

**AWS GUI Steps**:

1. **Prepare Training Script** (in Studio):
```python
# training_script.py
import argparse
import json
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--train-data', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load dataset
    dataset = load_dataset('json', data_files=f"{args.train_data}/train.jsonl")
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples['prompt'] + examples['completion'], 
                        truncation=True, 
                        max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

if __name__ == '__main__':
    main()
```

2. **Navigate to SageMaker Console** (Updated UI)
   - **Option A - Classic Console**:
     - Go to SageMaker Console → "Training" → "Training jobs"
     - Click "Create training job"
   - **Option B - SageMaker Studio** (Recommended):
     - Open SageMaker Studio
     - Click "Home" icon → "Deployments" → "Training" → "Training jobs"
     - Click "Create training job"

3. **Configure Training Job** (Reorganized UI)
   - **Job settings**:
     - **Job name**: "llama-finetuning-job-001-[timestamp]"
     - **IAM role**: Select "SageMakerTrainingRole" or create new
   - **Algorithm options**:
     - Choose **"Your own algorithm container"** for custom scripts
     - Or **"Built-in algorithm"** for AWS pre-built algorithms

4. **Configure Algorithm** (Updated Container)
   - **Container definition**:
     - **Image URI** (Updated for 2024): 
       - For us-east-1: `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04`
       - For other regions, replace region code in URI
     - **Input mode**: File (default) or FastFile (for large datasets)
     - **Entry point**: training_script.py
   - **Hyperparameters** (Key-value pairs):
     - model-name: meta-llama/Llama-2-7b-hf
     - epochs: 3
     - batch-size: 4
     - learning-rate: 0.00002
   - **Metric definitions** (Optional but recommended):
     - Add custom metrics to track in CloudWatch
     - Example: Name: "train:loss", Regex: "loss: ([0-9\\.]+)"

5. **Configure Input Data** (New Interface)
   - Click **"Add channel"** button
   - **Channel configuration**:
     - **Channel name**: "training" (or "train")
     - **Data source**: S3
     - **S3 data type**: S3Prefix
     - **S3 location**: s3://ml-training-data-[account]-[region]/training-data/processed/
     - **Content type**: application/jsonl or application/json
     - **Compression type**: None (or Gzip if compressed)
     - **Record wrapper**: None
   - Add additional channels for validation data if needed:
     - Click "Add channel" again
     - Channel name: "validation"
     - Configure similar to training channel

6. **Configure Output** (Enhanced Options)
   - **S3 output path**: s3://ml-model-artifacts-[account]-[region]/sagemaker-models/
   - **Encryption**: 
     - Enable KMS encryption (recommended for production)
     - Select KMS key or use default
   - **Output compression**: Enable to reduce storage costs

7. **Configure Resources** (Updated Instance Types)
   - **Instance configuration**:
     - **Instance type**: 
       - **ml.g5.xlarge** (newer, cost-effective, 1 GPU)
       - **ml.g5.2xlarge** (2 GPUs, recommended for medium models)
       - **ml.g5.12xlarge** (4 GPUs, for large models)
       - **ml.p4d.24xlarge** (8 GPUs, for distributed training)
     - **Instance count**: 
       - 1 for single-node training
       - 2+ for distributed training
     - **Volume size**: 100-500 GB (depends on model size)
       - 100 GB for models <7B parameters
       - 200-500 GB for larger models
   - **Additional storage** (Optional):
     - Enable for very large datasets

8. **Configure Stopping Condition**
   - **Max runtime**: 86400 seconds (24 hours) recommended
   - **Max wait time**: For spot instances (if using managed spot training)

9. **Configure VPC** (Optional but recommended for security)
   - **Enable network isolation**: Check for maximum security
   - **VPC**: Select "ml-training-vpc"
   - **Subnets**: Select private subnets (at least 2 for HA)
   - **Security groups**: Select training security group
   - Note: VPC configuration may increase training time slightly

10. **Configure Checkpointing** (New Feature)
    - **Enable checkpointing**: Recommended for long training jobs
    - **S3 checkpoint location**: s3://ml-model-artifacts-.../checkpoints/
    - Allows resuming training if interrupted

11. **Configure Managed Spot Training** (Cost Optimization)
    - **Enable managed spot training**: Can save up to 90% on costs
    - **Max wait time**: 86400 seconds (24 hours)
    - **Checkpoint S3 URI**: Required for spot training
    - Note: Training may be interrupted and resumed

12. **Add Tags** (For Cost Tracking)
    - Add tags to organize and track costs:
      - Key: "Project", Value: "ModelFineTuning"
      - Key: "Environment", Value: "Development"
      - Key: "CostCenter", Value: "ML-Team"

13. **Review and Create**
    - Review all configurations carefully
    - Verify S3 paths, IAM roles, and instance types
    - Click **"Create training job"**
    - Job will appear in training jobs list

14. **Monitor Training Progress** (Enhanced Monitoring)
    - **In SageMaker Console**:
      - Go to "Training jobs" → Click on your job name
      - View status: InProgress → Completed/Failed/Stopped
      - **Job details** tab shows configuration
      - **Monitor** tab shows:
        - Instance metrics (CPU, GPU, memory)
        - Custom metrics (loss, accuracy)
        - Training time and cost
    - **In CloudWatch**:
      - Click "View logs" link
      - Navigate to log streams for detailed output
      - Set up log insights queries for analysis
    - **Training typically takes**:
      - 2-8 hours for 7B models
      - 8-24 hours for 13B+ models
      - Depends on dataset size and epochs

15. **Download Model Artifacts**
    - Once training completes, artifacts are saved to S3
    - Navigate to S3 output path
    - Download model.tar.gz file
    - Extract to view model files and checkpoints

### Phase 5: Model Evaluation

#### Component 5.1: Evaluate Fine-Tuned Models

**Purpose**: Measure model performance and compare with baseline.

**AWS GUI Steps**:

1. **Create Evaluation Notebook in Studio**:
```python
import boto3
import json
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Initialize clients
bedrock_runtime = boto3.client('bedrock-runtime')
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Load test dataset
test_data = load_dataset('json', data_files='test.jsonl')['train']

def evaluate_bedrock_model(model_id, test_data):
    """Evaluate Bedrock custom model"""
    predictions = []
    ground_truth = []
    
    for item in test_data:
        # Call Bedrock model
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "messages": [{
                "role": "user",
                "content": item['prompt']
            }]
        })
        
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        prediction = response_body['content'][0]['text']
        
        predictions.append(prediction)
        ground_truth.append(item['completion'])
    
    # Calculate metrics
    # (Add your specific metrics here)
    
    return {
        'predictions': predictions,
        'ground_truth': ground_truth,
        'accuracy': calculate_accuracy(predictions, ground_truth)
    }

def evaluate_sagemaker_model(endpoint_name, test_data):
    """Evaluate SageMaker endpoint"""
    predictions = []
    ground_truth = []
    
    for item in test_data:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps({'inputs': item['prompt']})
        )
        
        prediction = json.loads(response['Body'].read())
        predictions.append(prediction)
        ground_truth.append(item['completion'])
    
    return {
        'predictions': predictions,
        'ground_truth': ground_truth,
        'accuracy': calculate_accuracy(predictions, ground_truth)
    }

# Run evaluations
bedrock_results = evaluate_bedrock_model('your-custom-model-id', test_data)
print(f"Bedrock Model Accuracy: {bedrock_results['accuracy']}")

# Generate evaluation report
report = {
    'model_id': 'custom-model-v1',
    'test_samples': len(test_data),
    'accuracy': bedrock_results['accuracy'],
    'timestamp': str(datetime.now())
}

# Save report to S3
s3 = boto3.client('s3')
s3.put_object(
    Bucket='ml-evaluation-results-[account]-[region]',
    Key='evaluation/reports/model-v1-evaluation.json',
    Body=json.dumps(report, indent=2)
)
```

2. **Create Evaluation Dashboard in CloudWatch**
   - Navigate to CloudWatch → Dashboards
   - Create dashboard: "Model-Evaluation-Dashboard"
   - Add custom metrics for accuracy, F1, latency

### Phase 6: Model Deployment

#### Component 6.1: Deploy to SageMaker Endpoint

**AWS GUI Steps** (Updated for 2024):

1. **Navigate to SageMaker Console**
   - Go to SageMaker Console → "Inference" → "Models"
   - Click "Create model" button

2. **Configure Model** (Simplified UI)
   - **Model settings**:
     - **Model name**: "finetuned-llama-v1-[timestamp]"
     - **IAM role**: Select "SageMakerDeploymentRole"
       - Role must have permissions for S3, ECR, and CloudWatch
   - **Container definition**:
     - **Provide model artifacts**: Yes
     - **Location of inference code image**: 
       - Use same container as training or inference-specific container
       - Example: `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04`
     - **Location of model artifacts**: 
       - S3 URI: s3://ml-model-artifacts-[account]-[region]/sagemaker-models/[job-name]/output/model.tar.gz
       - Browse S3 to select the model.tar.gz file
   - **Network configuration** (Optional):
     - Enable VPC if required for security
   - Click "Create model"

3. **Create Endpoint Configuration** (Enhanced Options)
   - Navigate to "Endpoint configurations" → Click "Create endpoint configuration"
   - **Endpoint configuration settings**:
     - **Name**: "finetuned-llama-config-v1"
     - **Encryption**: Enable KMS encryption (optional)
   - **Production variants**:
     - Click "Add model"
     - **Model**: Select "finetuned-llama-v1"
     - **Variant name**: "AllTraffic" (or "VariantA" for A/B testing)
     - **Instance type**: 
       - **ml.g5.xlarge** (recommended, 1 GPU, cost-effective)
       - **ml.g5.2xlarge** (2 GPUs, higher throughput)
       - **ml.inf2.xlarge** (AWS Inferentia, cost-optimized)
     - **Initial instance count**: 1 (scale up based on load)
     - **Initial variant weight**: 1 (for traffic distribution)
     - **Accelerator type**: None (GPU already in instance)
   - **Data capture** (New Feature - for Model Monitor):
     - **Enable data capture**: Yes (recommended for monitoring)
     - **Sampling percentage**: 100% for development, 10-20% for production
     - **Destination S3 URI**: s3://ml-model-artifacts-.../data-capture/
   - **Auto-scaling** (Configure later after endpoint creation)
   - Click "Create endpoint configuration"

4. **Create Endpoint** (Updated Process)
   - Navigate to "Endpoints" → Click "Create endpoint"
   - **Endpoint settings**:
     - **Endpoint name**: "finetuned-llama-endpoint-v1"
     - **Attach endpoint configuration**: 
       - Select "Use an existing endpoint configuration"
       - Choose "finetuned-llama-config-v1"
   - **Tags** (Optional):
     - Add tags for organization and cost tracking
   - Click "Create endpoint"
   - **Deployment time**: 
     - Status will show: Creating → InService
     - Typically takes 5-15 minutes
     - Monitor progress in endpoint details page

5. **Configure Auto-Scaling** (Post-Deployment)
   - Once endpoint is InService, click on endpoint name
   - Go to "Endpoint runtime settings" tab
   - Click "Configure auto scaling"
   - **Auto-scaling configuration**:
     - **Variant name**: Select your variant
     - **Minimum instance count**: 1
     - **Maximum instance count**: 5-10 (based on budget)
     - **Target metric**: 
       - SageMakerVariantInvocationsPerInstance
       - Target value: 1000-5000 (adjust based on latency requirements)
     - **Scale-in cool down**: 300 seconds
     - **Scale-out cool down**: 60 seconds
   - Click "Save"

6. **Test Endpoint** (Multiple Methods)

   **Method A - Using Python SDK**:
   ```python
   import boto3
   import json
   
   runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
   
   # Prepare input
   payload = {
       'inputs': 'Your test prompt here',
       'parameters': {
           'max_new_tokens': 200,
           'temperature': 0.7,
           'top_p': 0.9
       }
   }
   
   # Invoke endpoint
   response = runtime.invoke_endpoint(
       EndpointName='finetuned-llama-endpoint-v1',
       ContentType='application/json',
       Body=json.dumps(payload)
   )
   
   # Parse response
   result = json.loads(response['Body'].read().decode())
   print("Model response:", result)
   print("Latency:", response['ResponseMetadata']['HTTPHeaders'].get('x-amzn-invoked-production-variant'))
   ```

   **Method B - Using AWS CLI**:
   ```bash
   aws sagemaker-runtime invoke-endpoint \
       --endpoint-name finetuned-llama-endpoint-v1 \
       --content-type application/json \
       --body '{"inputs": "Your test prompt"}' \
       output.json
   
   cat output.json
   ```

   **Method C - Using SageMaker Studio**:
   - In Studio, create new notebook
   - Use boto3 code above
   - Run cells to test endpoint

7. **Monitor Endpoint Performance**
   - Go to endpoint details page
   - Click "Monitor" tab
   - View metrics:
     - Invocations per minute
     - Model latency (P50, P90, P99)
     - CPU/GPU utilization
     - Memory usage
   - Set up CloudWatch alarms for:
     - High latency (>2 seconds)
     - High error rate (>1%)
     - Instance health

8. **Verify Endpoint Health**
   ```python
   import boto3
   
   sagemaker = boto3.client('sagemaker')
   
   # Describe endpoint
   response = sagemaker.describe_endpoint(
       EndpointName='finetuned-llama-endpoint-v1'
   )
   
   print("Endpoint Status:", response['EndpointStatus'])
   print("Instance Count:", response['ProductionVariants'][0]['CurrentInstanceCount'])
   print("Desired Instance Count:", response['ProductionVariants'][0]['DesiredInstanceCount'])
   ```

**Important Notes**:
- Endpoints incur costs while running (even with no traffic)
- Use auto-scaling to optimize costs
- Enable data capture for model monitoring
- Test thoroughly before production deployment
- Consider using serverless inference for sporadic traffic
- Monitor costs in AWS Cost Explorer

### Phase 7: Model Registry and Versioning

**AWS GUI Steps**:

1. Navigate to SageMaker → Model Registry
2. Create model package group: "production-models"
3. Register model with metadata
4. Approve for production deployment
5. Track lineage and versions

### Phase 8: Monitoring and Continuous Improvement

**AWS GUI Steps**:

1. Enable SageMaker Model Monitor
2. Configure data quality monitoring
3. Set up drift detection
4. Create CloudWatch alarms
5. Implement retraining triggers

## Production Deployment Checklist

- [ ] Training data prepared and validated
- [ ] Fine-tuning completed successfully
- [ ] Model evaluated and meets performance criteria
- [ ] Model registered in Model Registry
- [ ] Endpoint deployed and tested
- [ ] Monitoring enabled
- [ ] Cost budgets configured
- [ ] Security controls implemented
- [ ] Documentation completed

## Troubleshooting Common UI Issues

### Issue 1: Bedrock Custom Models Not Visible
**Symptoms**: Cannot find "Custom models" option in Bedrock console

**Solutions**:
- Ensure you've requested model access: Bedrock → "Model access" → "Manage model access"
- Some models require approval (can take 24-48 hours)
- Check you're in a supported region (us-east-1, us-west-2 recommended)
- Verify IAM permissions include `bedrock:*` actions

### Issue 2: Training Job Fails Immediately
**Symptoms**: SageMaker training job fails within seconds

**Solutions**:
- **Check IAM role permissions**:
  - Role must have S3 read/write access
  - CloudWatch Logs write access
  - ECR pull permissions for container images
- **Verify data format**:
  - JSONL files must have one JSON object per line
  - No trailing commas or invalid JSON
  - Check file encoding (UTF-8 required)
- **Validate S3 paths**:
  - Ensure bucket exists and is accessible
  - Check bucket is in same region as training job
  - Verify file paths are correct (no typos)
- **Review CloudWatch logs**:
  - Go to training job → "Monitor" → "View logs"
  - Look for error messages in log streams

### Issue 3: Endpoint Creation Stuck or Fails
**Symptoms**: Endpoint stays in "Creating" status or fails to deploy

**Solutions**:
- **Check VPC configuration** (if using VPC):
  - Ensure subnets have available IP addresses
  - Security groups allow necessary traffic
  - NAT Gateway configured for internet access
- **Verify instance availability**:
  - Some GPU instances have limited availability
  - Try different instance type or region
  - Check service quotas: AWS Console → Service Quotas → SageMaker
- **Validate model artifacts**:
  - Ensure model.tar.gz is properly formatted
  - Check file size (not corrupted)
  - Verify S3 path is accessible
- **Review endpoint logs**:
  - CloudWatch Logs → /aws/sagemaker/Endpoints/[endpoint-name]

### Issue 4: Data Format Errors in Bedrock
**Symptoms**: "Invalid data format" error during fine-tuning job creation

**Solutions**:
- **Verify JSONL format**:
  ```python
  # Correct format
  {"prompt": "Question: What is AI?", "completion": " AI is artificial intelligence."}
  {"prompt": "Question: What is ML?", "completion": " ML is machine learning."}
  
  # Each line must be valid JSON
  # No commas between lines
  # Space before completion text is recommended
  ```
- **Check file encoding**: Must be UTF-8
- **Validate with Python**:
  ```python
  import jsonlines
  
  with jsonlines.open('train.jsonl') as reader:
      for i, obj in enumerate(reader):
          if 'prompt' not in obj or 'completion' not in obj:
              print(f"Line {i+1} missing required fields")
          if not isinstance(obj['prompt'], str) or not isinstance(obj['completion'], str):
              print(f"Line {i+1} has non-string values")
  ```
- **Minimum dataset size**: At least 32 examples required
- **Maximum file size**: 10 GB limit

### Issue 5: High Training Costs
**Symptoms**: Unexpected high costs for training jobs

**Solutions**:
- **Use managed spot training**:
  - Can save up to 90% on training costs
  - Enable in training job configuration
  - Requires checkpointing enabled
- **Optimize instance selection**:
  - Start with smaller instances (ml.g5.xlarge)
  - Scale up only if needed
  - Use CPU instances for small models
- **Set stopping conditions**:
  - Configure max runtime to prevent runaway jobs
  - Use early stopping in hyperparameters
- **Monitor with AWS Budgets**:
  - Set up budget alerts
  - Track costs by tags
- **Use Bedrock for simpler use cases**:
  - Managed service with predictable pricing
  - No infrastructure management

### Issue 6: Slow Inference Latency
**Symptoms**: Endpoint responses take >5 seconds

**Solutions**:
- **Optimize instance type**:
  - Use GPU instances (ml.g5.x) for large models
  - Consider AWS Inferentia (ml.inf2.x) for cost-optimized inference
- **Enable model compilation**:
  - Use SageMaker Neo to optimize model
  - Can reduce latency by 2-3x
- **Implement request batching**:
  - Batch multiple requests together
  - Configure in endpoint settings
- **Use model quantization**:
  - Reduce model precision (FP16 or INT8)
  - Significantly reduces latency
- **Check network configuration**:
  - VPC endpoints can add latency
  - Consider direct internet access for lower latency

### Issue 7: Model Access Denied in Bedrock
**Symptoms**: "Access denied" when trying to use base models

**Solutions**:
- Request model access: Bedrock → "Model access" → Enable models
- Wait for approval (instant for most models, up to 48 hours for some)
- Check IAM permissions include:
  - `bedrock:InvokeModel`
  - `bedrock:CreateModelCustomizationJob`
  - `bedrock:GetModelCustomizationJob`
- Verify you're in correct AWS region
- Some models have usage restrictions (check model cards)

### Issue 8: SageMaker Studio Won't Open
**Symptoms**: Studio fails to load or shows errors

**Solutions**:
- **Clear browser cache**: Hard refresh (Ctrl+Shift+R)
- **Check domain status**: SageMaker → Studio → Domain status should be "InService"
- **Verify user profile**: Ensure user profile is created and active
- **Check IAM permissions**: User needs SageMaker Studio access
- **Try different browser**: Chrome or Firefox recommended
- **Check VPC configuration**: If using VPC, ensure proper networking

### Issue 9: Out of Memory Errors During Training
**Symptoms**: Training fails with OOM (Out of Memory) errors

**Solutions**:
- **Reduce batch size**: Lower per_device_train_batch_size
- **Use gradient accumulation**: Simulate larger batches
- **Enable gradient checkpointing**: Trades compute for memory
- **Use larger instance**: Upgrade to instance with more GPU memory
- **Reduce sequence length**: Truncate inputs to shorter length
- **Use mixed precision training**: Enable FP16 or BF16

### Issue 10: Cannot Find Training Logs
**Symptoms**: CloudWatch logs not appearing for training job

**Solutions**:
- Wait 2-3 minutes after job starts (logs have delay)
- Check IAM role has CloudWatch Logs write permissions
- Navigate to: CloudWatch → Logs → Log groups → /aws/sagemaker/TrainingJobs
- Look for log stream with your job name
- If still missing, check training job failed immediately (see Issue 2)

### Getting Additional Help

**AWS Support Resources**:
- AWS Support Center: https://console.aws.amazon.com/support
- SageMaker Documentation: https://docs.aws.amazon.com/sagemaker
- Bedrock Documentation: https://docs.aws.amazon.com/bedrock
- AWS re:Post Community: https://repost.aws
- GitHub Issues: Check AWS SDK repositories

**Best Practices for Troubleshooting**:
1. Always check CloudWatch logs first
2. Verify IAM permissions for all services
3. Test with minimal configuration first
4. Use AWS CLI to verify resource states
5. Enable detailed logging and monitoring
6. Keep track of all resource ARNs and names
7. Document your configuration for reproducibility

---

**Document Version**: 2.0  
**Last Updated**: 2024-11-24  
**Status**: Production Ready - Updated for Latest AWS Console UI
