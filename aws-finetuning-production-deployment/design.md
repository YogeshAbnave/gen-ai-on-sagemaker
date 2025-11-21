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
2. **Install dependencies**:
```python
!pip install datasets pandas jsonlines
```

3. **Load and format data**:
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

4. **Upload to S3**:
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

**AWS GUI Steps**:

1. **Navigate to Bedrock Console**
   - Search for "Bedrock"
   - Click "Amazon Bedrock"

2. **Access Custom Models**
   - Click "Custom models" in left sidebar
   - Click "Create custom model"

3. **Select Base Model**
   - **Model name**: "custom-domain-model-v1"
   - **Base model**: Select "Claude 3 Haiku" or "Titan Text G1 - Express"
   - Click "Next"

4. **Configure Training Data**
   - **Training data location**: Browse S3 → select "s3://ml-training-data-.../train.jsonl"
   - **Validation data location**: Browse S3 → select "s3://ml-training-data-.../validation.jsonl"
   - **Output data location**: Browse S3 → select "s3://ml-model-artifacts-.../bedrock-models/"
   - Click "Next"

5. **Configure Hyperparameters**
   - **Epochs**: 3-5 (start with 3)
   - **Batch size**: 8-16
   - **Learning rate**: 0.00001-0.0001
   - **Warmup steps**: 100
   - Click "Next"

6. **Review and Create**
   - Review all settings
   - Click "Create custom model"
   - Training will take 2-6 hours depending on data size

7. **Monitor Training**
   - Go to "Custom models" → Click on your model
   - View training progress and metrics
   - Check CloudWatch logs for detailed information

8. **Provision Custom Model**
   - Once training completes, click "Purchase provisioned throughput"
   - **Provisioned throughput name**: "custom-model-endpoint"
   - **Model units**: Start with 1 (can scale up)
   - Click "Purchase"
   - Wait 10-15 minutes for provisioning

9. **Test Custom Model**
   - Go to "Playgrounds" → "Chat"
   - Select your custom model from dropdown
   - Test with sample prompts
   - Compare responses with base model

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

2. **Navigate to SageMaker Console**
   - Click "Training" → "Training jobs"
   - Click "Create training job"

3. **Configure Training Job**
   - **Job name**: "llama-finetuning-job-001"
   - **IAM role**: Select "SageMakerTrainingRole"
   - **Algorithm source**: Your own algorithm container or script mode

4. **Configure Algorithm**
   - **Container**: Use HuggingFace container
   - **Image URI**: `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04`
   - **Entry point**: training_script.py
   - **Hyperparameters**:
     - epochs: 3
     - batch-size: 4
     - learning-rate: 0.00002

5. **Configure Input Data**
   - **Channel name**: train
   - **S3 location**: s3://ml-training-data-.../training-data/processed/
   - **Content type**: application/json

6. **Configure Output**
   - **S3 output path**: s3://ml-model-artifacts-.../sagemaker-models/

7. **Configure Resources**
   - **Instance type**: ml.g5.2xlarge (or ml.p3.2xlarge)
   - **Instance count**: 1 (or more for distributed training)
   - **Volume size**: 100 GB

8. **Configure Stopping Condition**
   - **Max runtime**: 24 hours

9. **Configure VPC** (Optional but recommended)
   - **VPC**: Select "ml-training-vpc"
   - **Subnets**: Select private subnets
   - **Security groups**: Select training security group

10. **Create Training Job**
    - Click "Create training job"
    - Monitor progress in console
    - View logs in CloudWatch

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

**AWS GUI Steps**:

1. **Navigate to SageMaker Console**
   - Click "Inference" → "Models"
   - Click "Create model"

2. **Configure Model**
   - **Model name**: "finetuned-llama-v1"
   - **IAM role**: SageMakerDeploymentRole
   - **Container**: Same as training
   - **Model artifacts**: s3://ml-model-artifacts-.../model.tar.gz

3. **Create Endpoint Configuration**
   - Click "Endpoint configurations" → "Create"
   - **Name**: "finetuned-llama-config"
   - **Production variants**:
     - Instance type: ml.g5.xlarge
     - Initial instance count: 1
     - Initial weight: 1

4. **Create Endpoint**
   - Click "Endpoints" → "Create endpoint"
   - **Name**: "finetuned-llama-endpoint"
   - **Endpoint configuration**: Select "finetuned-llama-config"
   - Click "Create endpoint"
   - Wait 5-10 minutes for deployment

5. **Test Endpoint**
```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')

response = runtime.invoke_endpoint(
    EndpointName='finetuned-llama-endpoint',
    ContentType='application/json',
    Body=json.dumps({'inputs': 'Your test prompt here'})
)

result = json.loads(response['Body'].read())
print(result)
```

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

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-01  
**Status**: Production Ready
