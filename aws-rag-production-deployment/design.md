# Design Document: Production-Grade RAG Application on AWS

## Overview

This design document provides comprehensive AWS GUI step-by-step instructions for deploying a production-grade Retrieval Augmented Generation (RAG) application using Amazon SageMaker AI, Amazon Bedrock, and Amazon OpenSearch Service. The design follows AWS Well-Architected Framework principles and real-world best practices for security, scalability, reliability, performance efficiency, and cost optimization.

The RAG system architecture consists of:
- **Data Layer**: S3 for document storage, OpenSearch for vector database
- **Compute Layer**: SageMaker for embedding models, Bedrock for foundation models
- **Application Layer**: Lambda functions, API Gateway for programmatic access
- **Security Layer**: IAM roles, VPC isolation, KMS encryption, Bedrock Guardrails
- **Monitoring Layer**: CloudWatch for metrics, logs, and alarms

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
│  │  │  Public Subnet   │         │  Private Subnet  │            │ │
│  │  │  (10.0.1.0/24)   │         │  (10.0.10.0/24)  │            │ │
│  │  │                  │         │                  │            │ │
│  │  │  ┌────────────┐  │         │  ┌────────────┐ │            │ │
│  │  │  │ NAT Gateway│  │         │  │ SageMaker  │ │            │ │
│  │  │  └────────────┘  │         │  │  Studio    │ │            │ │
│  │  │                  │         │  └────────────┘ │            │ │
│  │  └──────────────────┘         │                  │            │ │
│  │                               │  ┌────────────┐ │            │ │
│  │                               │  │ OpenSearch │ │            │ │
│  │                               │  │  Domain    │ │            │ │
│  │                               │  └────────────┘ │            │ │
│  │                               └──────────────────┘            │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  S3 Buckets  │  │   Bedrock    │  │  CloudWatch  │              │
│  │  (Regional)  │  │  (Regional)  │  │  (Regional)  │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ API Gateway  │  │    Lambda    │  │     IAM      │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```


### Data Flow Architecture

```
User Query → API Gateway → Lambda → [Query Embedding via SageMaker]
                                   ↓
                          [Vector Search in OpenSearch]
                                   ↓
                          [Retrieve Top-K Documents]
                                   ↓
                          [Construct Prompt with Context]
                                   ↓
                          [Generate Response via Bedrock]
                                   ↓
                          [Apply Guardrails Filter]
                                   ↓
                          Response ← Lambda ← API Gateway ← User
```

## Components and Interfaces

### Phase 1: Foundation Infrastructure Setup (Prerequisites)

This phase establishes the foundational AWS infrastructure required for the RAG application.

#### Component 1.1: VPC and Network Configuration

**Purpose**: Create an isolated network environment with public and private subnets for secure component deployment.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to VPC Service**
   - Log into AWS Console (https://console.aws.amazon.com)
   - In the search bar at the top, type "VPC"
   - Click on "VPC" from the dropdown results

2. **Launch VPC Creation Wizard**
   - Click the orange "Create VPC" button in the top-right
   - Select "VPC and more" option (this creates VPC with subnets, route tables, and gateways automatically)

3. **Configure VPC Settings**
   - **Name tag auto-generation**: Enter "rag-production-vpc"
   - **IPv4 CIDR block**: Enter "10.0.0.0/16"
   - **IPv6 CIDR block**: Select "No IPv6 CIDR block"
   - **Tenancy**: Select "Default"

4. **Configure Subnets**
   - **Number of Availability Zones**: Select "2"
   - **Number of public subnets**: Enter "2"
   - **Number of private subnets**: Enter "2"
   - **Public subnet CIDR blocks**: 
     - AZ1: 10.0.1.0/24
     - AZ2: 10.0.2.0/24
   - **Private subnet CIDR blocks**:
     - AZ1: 10.0.10.0/24
     - AZ2: 10.0.11.0/24

5. **Configure NAT Gateways**
   - **NAT gateways**: Select "1 per AZ" (for high availability)
   - **VPC endpoints**: Select "None" (we'll add specific endpoints later)

6. **Review and Create**
   - Scroll down and review the preview diagram
   - Click "Create VPC" button
   - Wait 2-3 minutes for creation to complete
   - Click "View VPC" to see your created VPC

7. **Note Important IDs** (save these for later use)
   - VPC ID (e.g., vpc-0abc123def456)
   - Private Subnet IDs (e.g., subnet-0abc123, subnet-0def456)
   - Security Group ID (default security group)


#### Component 1.2: Security Groups Configuration

**Purpose**: Create firewall rules to control traffic between RAG components.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to Security Groups**
   - In VPC Console, click "Security Groups" in the left sidebar
   - Click "Create security group" button

2. **Create SageMaker Security Group**
   - **Security group name**: "rag-sagemaker-sg"
   - **Description**: "Security group for SageMaker Studio and endpoints"
   - **VPC**: Select your "rag-production-vpc"
   
3. **Configure Inbound Rules for SageMaker**
   - Click "Add rule" under Inbound rules
   - **Type**: HTTPS
   - **Protocol**: TCP
   - **Port range**: 443
   - **Source**: Custom - Select the same security group (self-referencing)
   - Click "Add rule" again
   - **Type**: All TCP
   - **Protocol**: TCP
   - **Port range**: 0-65535
   - **Source**: Custom - Select the same security group
   
4. **Configure Outbound Rules for SageMaker**
   - Leave default (All traffic to 0.0.0.0/0)
   - Click "Create security group"

5. **Create OpenSearch Security Group**
   - Click "Create security group" again
   - **Security group name**: "rag-opensearch-sg"
   - **Description**: "Security group for OpenSearch domain"
   - **VPC**: Select your "rag-production-vpc"

6. **Configure Inbound Rules for OpenSearch**
   - Click "Add rule"
   - **Type**: HTTPS
   - **Protocol**: TCP
   - **Port range**: 443
   - **Source**: Custom - Select "rag-sagemaker-sg" (allows SageMaker to access OpenSearch)
   - Click "Create security group"

7. **Note Security Group IDs** (save for later)
   - SageMaker SG ID (e.g., sg-0abc123)
   - OpenSearch SG ID (e.g., sg-0def456)

#### Component 1.3: S3 Bucket Creation

**Purpose**: Create S3 buckets for storing documents, model artifacts, and logs.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to S3 Service**
   - In AWS Console search bar, type "S3"
   - Click on "S3" from results

2. **Create Documents Bucket**
   - Click "Create bucket" button
   - **Bucket name**: "rag-documents-[your-account-id]-[region]" (must be globally unique)
     - Example: "rag-documents-123456789012-us-east-1"
   - **AWS Region**: Select your preferred region (e.g., US East N. Virginia)
   - **Object Ownership**: Keep "ACLs disabled (recommended)"
   
3. **Configure Bucket Settings**
   - **Block Public Access settings**: Keep all boxes checked (block all public access)
   - **Bucket Versioning**: Select "Enable"
   - **Default encryption**: 
     - Select "Server-side encryption with Amazon S3 managed keys (SSE-S3)"
     - Or select "Server-side encryption with AWS Key Management Service keys (SSE-KMS)" for enhanced security
   - Click "Create bucket"

4. **Create Model Artifacts Bucket**
   - Click "Create bucket" again
   - **Bucket name**: "rag-models-[your-account-id]-[region]"
   - Follow same settings as documents bucket
   - Click "Create bucket"

5. **Create Logs Bucket**
   - Click "Create bucket" again
   - **Bucket name**: "rag-logs-[your-account-id]-[region]"
   - Follow same settings as documents bucket
   - Click "Create bucket"

6. **Create Folder Structure in Documents Bucket**
   - Click on "rag-documents-..." bucket name
   - Click "Create folder" button
   - Create folders: "raw-documents", "processed-documents", "embeddings"
   - Click "Create folder" for each

7. **Note Bucket Names** (save for later)
   - Documents bucket name
   - Models bucket name
   - Logs bucket name


#### Component 1.4: IAM Roles and Policies

**Purpose**: Create IAM roles with appropriate permissions for SageMaker, Lambda, and other services.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to IAM Service**
   - In AWS Console search bar, type "IAM"
   - Click on "IAM" from results

2. **Create SageMaker Execution Role**
   - Click "Roles" in the left sidebar
   - Click "Create role" button
   - **Trusted entity type**: Select "AWS service"
   - **Use case**: Select "SageMaker" from dropdown
   - Select "SageMaker - Execution"
   - Click "Next"

3. **Attach Policies to SageMaker Role**
   - In the search box, search and select these policies:
     - ✓ AmazonSageMakerFullAccess
     - ✓ AmazonS3FullAccess (or create custom policy for specific buckets)
     - ✓ AmazonBedrockFullAccess
   - Click "Next"

4. **Name the SageMaker Role**
   - **Role name**: "RAGSageMakerExecutionRole"
   - **Description**: "Execution role for SageMaker Studio and endpoints in RAG application"
   - Click "Create role"

5. **Add Inline Policy for OpenSearch Access**
   - Click on the newly created "RAGSageMakerExecutionRole"
   - Click "Add permissions" dropdown → "Create inline policy"
   - Click "JSON" tab
   - Paste the following policy:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "es:ESHttpGet",
           "es:ESHttpPut",
           "es:ESHttpPost",
           "es:ESHttpDelete",
           "es:ESHttpHead"
         ],
         "Resource": "arn:aws:es:*:*:domain/rag-*/*"
       }
     ]
   }
   ```
   - Click "Next"
   - **Policy name**: "OpenSearchAccessPolicy"
   - Click "Create policy"

6. **Create Lambda Execution Role**
   - Click "Roles" in left sidebar
   - Click "Create role"
   - **Trusted entity type**: "AWS service"
   - **Use case**: Select "Lambda"
   - Click "Next"

7. **Attach Policies to Lambda Role**
   - Search and select:
     - ✓ AWSLambdaBasicExecutionRole
     - ✓ AmazonSageMakerFullAccess
     - ✓ AmazonBedrockFullAccess
   - Click "Next"
   - **Role name**: "RAGLambdaExecutionRole"
   - Click "Create role"

8. **Create OpenSearch Service Role**
   - Click "Create role"
   - **Trusted entity type**: "AWS service"
   - **Use case**: Select "OpenSearch Service"
   - Click "Next"
   - No additional policies needed
   - Click "Next"
   - **Role name**: "RAGOpenSearchServiceRole"
   - Click "Create role"

9. **Note Role ARNs** (save for later)
   - SageMaker Role ARN (e.g., arn:aws:iam::123456789012:role/RAGSageMakerExecutionRole)
   - Lambda Role ARN
   - OpenSearch Role ARN


### Phase 2: SageMaker Studio Setup

This phase sets up the development environment for building and testing RAG components.

#### Component 2.1: SageMaker Domain Creation

**Purpose**: Create a SageMaker Domain to host Studio environments for data scientists.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to SageMaker Service**
   - In AWS Console search bar, type "SageMaker"
   - Click on "Amazon SageMaker" from results

2. **Access SageMaker Studio**
   - In the left sidebar, click "Domains" under "Admin configurations"
   - Click "Create domain" button

3. **Choose Setup Method**
   - Select "Set up for single user (Quick setup)" for faster setup
   - OR select "Set up for organization" for production with multiple users
   - For production, choose "Set up for organization"
   - Click "Set up"

4. **Configure Domain Settings (Organization Setup)**
   - **Domain name**: "rag-production-domain"
   - **Authentication method**: Select "AWS Identity and Access Management (IAM)"

5. **Configure Network and Storage**
   - **VPC**: Select "rag-production-vpc"
   - **Subnets**: Select both private subnets (10.0.10.0/24 and 10.0.11.0/24)
   - **Security group(s)**: Select "rag-sagemaker-sg"
   - **VPC only mode**: Leave unchecked (allows internet access through NAT)

6. **Configure Execution Role**
   - **Execution role**: Select "Use existing role"
   - Select "RAGSageMakerExecutionRole" from dropdown

7. **Configure Studio Settings**
   - **Notebook sharing**: Enable if needed
   - **Default JupyterLab version**: Select latest version (e.g., JupyterLab 3.0)
   - Click "Next"

8. **Configure RStudio Settings** (Optional)
   - Skip this section if not using RStudio
   - Click "Next"

9. **Review and Create**
   - Review all settings
   - Click "Submit"
   - Wait 5-10 minutes for domain creation (status will show "InService")

10. **Note Domain Details** (save for later)
    - Domain ID (e.g., d-abc123def456)
    - Domain ARN

#### Component 2.2: User Profile Creation

**Purpose**: Create user profiles for data scientists to access Studio.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to User Profiles**
   - In SageMaker Console, click "Domains" in left sidebar
   - Click on your "rag-production-domain"
   - Click "User profiles" tab

2. **Create User Profile**
   - Click "Add user" button
   - **Name**: "data-scientist-1"
   - **Default execution role**: Select "RAGSageMakerExecutionRole"

3. **Configure Studio Settings**
   - **Canvas settings**: Leave default or disable if not using Canvas
   - **JupyterLab version**: Select latest
   - Click "Next"

4. **Configure Storage**
   - **EFS storage**: Leave default (5 GB)
   - Click "Submit"
   - Wait 2-3 minutes for user profile creation

5. **Launch Studio**
   - Click on the user profile name "data-scientist-1"
   - Click "Launch" dropdown → Select "Studio"
   - Wait 3-5 minutes for Studio to launch
   - Studio will open in a new browser tab


#### Component 2.3: Upload Workshop Notebooks

**Purpose**: Upload and organize the RAG workshop notebooks in Studio.

**AWS GUI Step-by-Step Instructions**:

1. **In SageMaker Studio Interface**
   - You should now be in the JupyterLab interface
   - Left sidebar shows file browser

2. **Clone Workshop Repository**
   - Click the "Git" menu at the top
   - Select "Clone a Repository"
   - Enter repository URL: `https://github.com/aws-samples/generative-ai-on-amazon-sagemaker.git`
   - Click "Clone"
   - Wait for cloning to complete

3. **Navigate to RAG Workshop**
   - In file browser, expand "generative-ai-on-amazon-sagemaker"
   - Navigate to: `workshops/building-rag-workflows-with-sagemaker-and-bedrock/`

4. **Install Prerequisites**
   - Open `00-00_prerequisites/prerequisites.ipynb`
   - Select kernel: "Python 3 (Data Science 3.0)"
   - Run all cells to install required packages
   - Wait for installation to complete (5-10 minutes)

5. **Organize Workspace**
   - Create a new folder called "rag-production" in the root
   - Copy relevant notebooks to this folder for easier access

### Phase 3: Embedding Model Deployment

This phase deploys an embedding model to SageMaker for converting text to vectors.

#### Component 3.1: Deploy Embedding Model via JumpStart

**Purpose**: Deploy a pre-trained embedding model using SageMaker JumpStart.

**AWS GUI Step-by-Step Instructions**:

1. **Access JumpStart in Studio**
   - In SageMaker Studio, click the "Home" icon (house) in left sidebar
   - Click "JumpStart" in the left navigation panel
   - Or click "JumpStart" from the launcher page

2. **Search for Embedding Models**
   - In the search bar, type "embedding" or "sentence-transformers"
   - Browse available models:
     - "GPT-J-6B Embedding" (larger, more accurate)
     - "all-MiniLM-L6-v2" (smaller, faster)
     - "BGE-large-en" (recommended for production)

3. **Select BGE-large-en Model**
   - Click on "BGE-large-en-v1.5" model card
   - Review model details, performance metrics

4. **Configure Deployment**
   - Click "Deploy" button
   - **Endpoint name**: "rag-embedding-endpoint"
   - **Instance type**: Select "ml.g4dn.xlarge" (GPU instance for better performance)
     - For cost optimization, can use "ml.m5.xlarge" (CPU)
   - **Instance count**: 1 (increase for higher throughput)
   - **Endpoint configuration name**: Leave default or enter "rag-embedding-config"

5. **Advanced Settings** (Optional)
   - Expand "Advanced settings"
   - **Data capture**: Enable for monitoring (optional)
   - **Auto-scaling**: Configure if needed (we'll set this up later)

6. **Deploy Model**
   - Click "Deploy" button
   - Deployment will take 5-10 minutes
   - Monitor status in "Endpoints" section

7. **Verify Deployment**
   - In Studio, click "Deployments" in left sidebar
   - Click "Endpoints"
   - Find "rag-embedding-endpoint"
   - Status should show "InService" (green)

8. **Test Embedding Endpoint**
   - Create a new notebook in Studio
   - Run this test code:
   ```python
   import boto3
   import json
   
   runtime = boto3.client('sagemaker-runtime')
   
   payload = {
       "text_inputs": ["This is a test sentence for embedding."]
   }
   
   response = runtime.invoke_endpoint(
       EndpointName='rag-embedding-endpoint',
       ContentType='application/json',
       Body=json.dumps(payload)
   )
   
   result = json.loads(response['Body'].read())
   print(f"Embedding dimension: {len(result['embedding'][0])}")
   print(f"First 5 values: {result['embedding'][0][:5]}")
   ```

9. **Note Endpoint Details** (save for later)
   - Endpoint name: rag-embedding-endpoint
   - Endpoint ARN
   - Model dimension (e.g., 1024 for BGE-large)


### Phase 4: Amazon OpenSearch Service Setup

This phase creates a vector database for storing and searching document embeddings.

#### Component 4.1: Create OpenSearch Domain

**Purpose**: Deploy a managed OpenSearch cluster with vector search capabilities.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to OpenSearch Service**
   - In AWS Console search bar, type "OpenSearch"
   - Click on "Amazon OpenSearch Service"

2. **Create Domain**
   - Click "Create domain" button

3. **Choose Deployment Type**
   - **Domain name**: "rag-vector-db"
   - **Deployment type**: Select "Production"
   - **Version**: Select latest OpenSearch version (e.g., "OpenSearch 2.11")
   - **Auto-Tune**: Enable (recommended for automatic performance optimization)

4. **Configure Data Nodes**
   - **Instance type**: Select "r6g.large.search" (memory-optimized for vector search)
     - For development: "t3.small.search"
     - For production: "r6g.xlarge.search" or larger
   - **Number of nodes**: 2 (for high availability)
   - **Availability Zones**: 2-AZ (for production)

5. **Configure Storage**
   - **EBS storage type**: "General Purpose (SSD) - gp3"
   - **EBS storage size per node**: 100 GB (adjust based on data volume)
   - **EBS IOPS**: 3000 (default for gp3)
   - **EBS throughput**: 125 MB/s

6. **Configure Dedicated Master Nodes** (Recommended for Production)
   - **Dedicated master nodes**: Enable
   - **Instance type**: "r6g.large.search"
   - **Number of master nodes**: 3 (for high availability)

7. **Configure Network**
   - **Network**: Select "VPC access (recommended)"
   - **VPC**: Select "rag-production-vpc"
   - **Subnets**: Select ONE private subnet (e.g., 10.0.10.0/24)
     - Note: OpenSearch requires single subnet selection
   - **Security groups**: Select "rag-opensearch-sg"

8. **Configure Fine-grained Access Control**
   - **Enable fine-grained access control**: Check this box
   - **Create master user**: Select this option
   - **Master username**: "admin"
   - **Master password**: Create a strong password (save this securely!)
   - **Confirm password**: Re-enter password

9. **Configure Access Policy**
   - **Domain access policy**: Select "Only use fine-grained access control"
   - This is more secure than IP-based policies

10. **Configure Encryption**
    - **Encryption at rest**: Enable
    - **KMS key**: Select "aws/es" (AWS managed key) or choose custom KMS key
    - **Node-to-node encryption**: Enable (checked by default)
    - **Require HTTPS**: Enable (checked by default)

11. **Configure Advanced Settings**
    - **Automated snapshots**: Start hour: 0 (midnight UTC)
    - **CloudWatch Logs**: Enable all three log types:
      - ✓ Audit logs
      - ✓ Error logs
      - ✓ Search slow logs
    - **Index slow logs**: Enable
    - **Application logs**: Enable

12. **Review and Create**
    - Review all settings carefully
    - Click "Create" button
    - Domain creation takes 15-30 minutes
    - Status will change from "Loading" → "Active"

13. **Note Domain Details** (save for later)
    - Domain endpoint (e.g., https://vpc-rag-vector-db-abc123.us-east-1.es.amazonaws.com)
    - Domain ARN
    - Master username and password

14. **Wait for Domain to be Active**
    - Refresh the page periodically
    - When status shows "Active" with green indicator, proceed to next step


#### Component 4.2: Configure OpenSearch for Vector Search

**Purpose**: Set up OpenSearch indices with k-NN plugin for vector similarity search.

**AWS GUI Step-by-Step Instructions**:

1. **Access OpenSearch Dashboards**
   - In OpenSearch Service console, click on "rag-vector-db" domain
   - Find "OpenSearch Dashboards URL" (e.g., https://vpc-rag-vector-db-abc123.us-east-1.es.amazonaws.com/_dashboards)
   - Click the URL (opens in new tab)

2. **Login to Dashboards**
   - **Username**: admin
   - **Password**: [your master password]
   - Click "Log in"

3. **Access Dev Tools Console**
   - Click the hamburger menu (☰) in top-left
   - Scroll down and click "Dev Tools"
   - You'll see a console interface for running queries

4. **Create k-NN Index for Vectors**
   - In the Dev Tools console, paste this command:
   ```json
   PUT /rag-documents-index
   {
     "settings": {
       "index": {
         "knn": true,
         "knn.algo_param.ef_search": 512,
         "number_of_shards": 2,
         "number_of_replicas": 1
       }
     },
     "mappings": {
       "properties": {
         "document_id": {
           "type": "keyword"
         },
         "content": {
           "type": "text"
         },
         "embedding": {
           "type": "knn_vector",
           "dimension": 1024,
           "method": {
             "name": "hnsw",
             "space_type": "cosinesimil",
             "engine": "nmslib",
             "parameters": {
               "ef_construction": 512,
               "m": 16
             }
           }
         },
         "metadata": {
           "type": "object",
           "properties": {
             "source": {"type": "keyword"},
             "page": {"type": "integer"},
             "timestamp": {"type": "date"}
           }
         }
       }
     }
   }
   ```
   - Click the green play button (▶) or press Ctrl+Enter
   - You should see a success response: `"acknowledged": true`

5. **Verify Index Creation**
   - Run this command to check the index:
   ```json
   GET /rag-documents-index
   ```
   - You should see the index settings and mappings

6. **Create Index Template for Future Indices** (Optional)
   - This allows automatic index creation with correct settings:
   ```json
   PUT /_index_template/rag-template
   {
     "index_patterns": ["rag-*"],
     "template": {
       "settings": {
         "index": {
           "knn": true,
           "knn.algo_param.ef_search": 512
         }
       }
     }
   }
   ```

7. **Test Vector Search Capability**
   - Insert a test document:
   ```json
   POST /rag-documents-index/_doc/test-1
   {
     "document_id": "test-1",
     "content": "This is a test document for RAG system.",
     "embedding": [0.1, 0.2, 0.3, ...],
     "metadata": {
       "source": "test",
       "page": 1,
       "timestamp": "2024-01-01T00:00:00Z"
     }
   }
   ```
   - Note: Replace [...] with actual 1024-dimension vector

8. **Configure Index Refresh Settings**
   ```json
   PUT /rag-documents-index/_settings
   {
     "index": {
       "refresh_interval": "30s"
     }
   }
   ```

9. **Note Index Details** (save for later)
   - Index name: rag-documents-index
   - Vector dimension: 1024
   - Space type: cosinesimil

### Phase 5: Amazon Bedrock Setup

This phase enables foundation models for text generation in the RAG pipeline.

#### Component 5.1: Enable Bedrock Model Access

**Purpose**: Request and enable access to foundation models in Amazon Bedrock.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to Amazon Bedrock**
   - In AWS Console search bar, type "Bedrock"
   - Click on "Amazon Bedrock"

2. **Access Model Access Page**
   - In the left sidebar, scroll down to "Bedrock configurations"
   - Click "Model access"

3. **Review Available Models**
   - You'll see a list of foundation models from different providers:
     - Amazon (Titan models)
     - Anthropic (Claude models)
     - AI21 Labs (Jurassic models)
     - Cohere (Command models)
     - Meta (Llama models)
     - Stability AI (Stable Diffusion)

4. **Request Model Access**
   - Click "Manage model access" button in top-right
   - For production RAG, select these models:
     - ✓ **Claude 3 Sonnet** (Anthropic) - Balanced performance and cost
     - ✓ **Claude 3.5 Sonnet** (Anthropic) - Best performance
     - ✓ **Titan Text G1 - Express** (Amazon) - Cost-effective option
     - ✓ **Titan Embeddings G1 - Text** (Amazon) - Alternative embedding model

5. **Review Terms and Conditions**
   - For each selected model, review the EULA
   - Some models require accepting terms

6. **Submit Access Request**
   - Click "Request model access" button at bottom
   - Most models are instantly available
   - Some may require approval (1-2 business days)

7. **Verify Model Access**
   - Wait a few seconds and refresh the page
   - Status should change to "Access granted" (green checkmark)
   - If status shows "In progress", wait and check back later

8. **Note Available Models** (save for later)
   - Model IDs (e.g., anthropic.claude-3-sonnet-20240229-v1:0)
   - Model pricing per 1K tokens


#### Component 5.2: Test Bedrock Models in Playground

**Purpose**: Verify model access and test generation capabilities.

**AWS GUI Step-by-Step Instructions**:

1. **Access Bedrock Playground**
   - In Bedrock console, click "Playgrounds" in left sidebar
   - Click "Chat" playground

2. **Select Model**
   - In the right panel under "Select model"
   - Choose "Anthropic" → "Claude 3 Sonnet"
   - Or choose "Claude 3.5 Sonnet" for better performance

3. **Configure Model Parameters**
   - **Temperature**: 0.7 (balance between creativity and consistency)
   - **Top P**: 0.9
   - **Top K**: 250
   - **Maximum length**: 2048 tokens
   - **Stop sequences**: Leave empty

4. **Test Model with Sample Prompt**
   - In the chat input box, enter:
   ```
   Based on the following context, answer the question.
   
   Context: Amazon SageMaker is a fully managed machine learning service. With SageMaker, data scientists and developers can quickly build and train machine learning models, and then directly deploy them into a production-ready hosted environment.
   
   Question: What is Amazon SageMaker?
   
   Answer:
   ```
   - Click "Run" button
   - Review the generated response

5. **Test with RAG-style Prompt**
   - Clear the chat and enter:
   ```
   You are a helpful AI assistant. Answer the question based only on the provided context. If the answer is not in the context, say "I don't have enough information to answer that."
   
   Context: [This will be replaced with retrieved documents]
   
   Question: [User question will go here]
   ```
   - This tests the prompt format for RAG

6. **Save Prompt Template**
   - Click "Save" button in top-right
   - **Name**: "RAG System Prompt"
   - **Description**: "Template for RAG question answering"
   - Click "Save"

7. **Test Different Models** (Optional)
   - Repeat tests with Titan Text Express
   - Compare response quality and latency
   - Choose the best model for your use case

### Phase 6: Document Ingestion and Processing

This phase implements the data pipeline for ingesting documents and creating embeddings.

#### Component 6.1: Upload Sample Documents to S3

**Purpose**: Prepare sample documents for the RAG system.

**AWS GUI Step-by-Step Instructions**:

1. **Prepare Sample Documents**
   - Create or gather 5-10 sample documents (PDF, TXT, or DOCX)
   - Topics should be related to your use case
   - For testing, you can use AWS documentation or public articles

2. **Navigate to S3 Console**
   - In AWS Console, go to S3 service
   - Click on "rag-documents-[account-id]-[region]" bucket

3. **Upload Documents**
   - Click on "raw-documents" folder
   - Click "Upload" button
   - Click "Add files" and select your documents
   - Or drag and drop files into the upload area

4. **Configure Upload Settings**
   - **Storage class**: Keep "Standard"
   - **Server-side encryption**: Already configured at bucket level
   - Click "Upload" button
   - Wait for upload to complete

5. **Verify Upload**
   - You should see all files listed in the raw-documents folder
   - Note the S3 URIs (e.g., s3://rag-documents-123456789012-us-east-1/raw-documents/doc1.pdf)

6. **Set Up Event Notification** (Optional - for automated processing)
   - Go back to bucket root
   - Click "Properties" tab
   - Scroll to "Event notifications"
   - Click "Create event notification"
   - **Name**: "document-upload-trigger"
   - **Event types**: Select "All object create events"
   - **Destination**: Lambda function (we'll create this later)
   - Click "Save changes"


#### Component 6.2: Create Document Processing Notebook

**Purpose**: Build a notebook to process documents, generate embeddings, and index in OpenSearch.

**AWS GUI Step-by-Step Instructions**:

1. **Return to SageMaker Studio**
   - Go back to your SageMaker Studio browser tab
   - Or launch Studio from SageMaker console

2. **Create New Notebook**
   - Click "File" → "New" → "Notebook"
   - **Kernel**: Select "Python 3 (Data Science 3.0)"
   - Click "Select"

3. **Install Required Libraries**
   - In the first cell, paste:
   ```python
   !pip install opensearch-py boto3 PyPDF2 langchain sentence-transformers -q
   ```
   - Run the cell (Shift+Enter)

4. **Import Libraries and Setup**
   - Create a new cell and paste:
   ```python
   import boto3
   import json
   from opensearchpy import OpenSearch, RequestsHttpConnection
   from requests_aws4auth import AWS4Auth
   import PyPDF2
   from io import BytesIO
   
   # Initialize AWS clients
   s3_client = boto3.client('s3')
   sagemaker_runtime = boto3.client('sagemaker-runtime')
   region = boto3.Session().region_name
   
   # Configuration
   BUCKET_NAME = 'rag-documents-[your-account-id]-[region]'
   EMBEDDING_ENDPOINT = 'rag-embedding-endpoint'
   OPENSEARCH_ENDPOINT = 'vpc-rag-vector-db-abc123.us-east-1.es.amazonaws.com'
   OPENSEARCH_INDEX = 'rag-documents-index'
   OPENSEARCH_USERNAME = 'admin'
   OPENSEARCH_PASSWORD = '[your-password]'
   ```
   - Replace placeholders with your actual values
   - Run the cell

5. **Create Document Processing Function**
   - New cell:
   ```python
   def extract_text_from_pdf(pdf_bytes):
       """Extract text from PDF bytes"""
       pdf_file = BytesIO(pdf_bytes)
       pdf_reader = PyPDF2.PdfReader(pdf_file)
       text = ""
       for page_num, page in enumerate(pdf_reader.pages):
           text += f"\n--- Page {page_num + 1} ---\n"
           text += page.extract_text()
       return text
   
   def chunk_text(text, chunk_size=500, overlap=50):
       """Split text into overlapping chunks"""
       words = text.split()
       chunks = []
       for i in range(0, len(words), chunk_size - overlap):
           chunk = ' '.join(words[i:i + chunk_size])
           chunks.append(chunk)
       return chunks
   ```
   - Run the cell

6. **Create Embedding Function**
   - New cell:
   ```python
   def get_embeddings(texts):
       """Get embeddings from SageMaker endpoint"""
       payload = {"text_inputs": texts}
       response = sagemaker_runtime.invoke_endpoint(
           EndpointName=EMBEDDING_ENDPOINT,
           ContentType='application/json',
           Body=json.dumps(payload)
       )
       result = json.loads(response['Body'].read())
       return result['embedding']
   ```
   - Run the cell

7. **Create OpenSearch Connection**
   - New cell:
   ```python
   # Connect to OpenSearch
   opensearch_client = OpenSearch(
       hosts=[{'host': OPENSEARCH_ENDPOINT, 'port': 443}],
       http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
       use_ssl=True,
       verify_certs=True,
       connection_class=RequestsHttpConnection
   )
   
   # Test connection
   print(opensearch_client.info())
   ```
   - Run the cell - should print cluster info

8. **Create Document Processing Pipeline**
   - New cell:
   ```python
   def process_and_index_document(bucket, key):
       """Process a document and index it in OpenSearch"""
       print(f"Processing: {key}")
       
       # Download document from S3
       response = s3_client.get_object(Bucket=bucket, Key=key)
       content = response['Body'].read()
       
       # Extract text
       if key.endswith('.pdf'):
           text = extract_text_from_pdf(content)
       elif key.endswith('.txt'):
           text = content.decode('utf-8')
       else:
           print(f"Unsupported file type: {key}")
           return
       
       # Chunk text
       chunks = chunk_text(text)
       print(f"Created {len(chunks)} chunks")
       
       # Process in batches
       batch_size = 10
       for i in range(0, len(chunks), batch_size):
           batch = chunks[i:i + batch_size]
           
           # Get embeddings
           embeddings = get_embeddings(batch)
           
           # Index in OpenSearch
           for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
               doc_id = f"{key.replace('/', '_')}_{i+j}"
               document = {
                   'document_id': doc_id,
                   'content': chunk,
                   'embedding': embedding,
                   'metadata': {
                       'source': key,
                       'chunk_index': i + j,
                       'timestamp': response['LastModified'].isoformat()
                   }
               }
               
               opensearch_client.index(
                   index=OPENSEARCH_INDEX,
                   id=doc_id,
                   body=document
               )
           
           print(f"Indexed batch {i//batch_size + 1}")
       
       print(f"Completed: {key}\n")
   ```
   - Run the cell

9. **Process All Documents**
   - New cell:
   ```python
   # List all documents in raw-documents folder
   response = s3_client.list_objects_v2(
       Bucket=BUCKET_NAME,
       Prefix='raw-documents/'
   )
   
   documents = [obj['Key'] for obj in response.get('Contents', []) 
                if not obj['Key'].endswith('/')]
   
   print(f"Found {len(documents)} documents to process\n")
   
   # Process each document
   for doc_key in documents:
       try:
           process_and_index_document(BUCKET_NAME, doc_key)
       except Exception as e:
           print(f"Error processing {doc_key}: {str(e)}\n")
   
   print("All documents processed!")
   ```
   - Run the cell - this will take several minutes

10. **Verify Indexing**
    - New cell:
    ```python
    # Check document count
    count = opensearch_client.count(index=OPENSEARCH_INDEX)
    print(f"Total documents in index: {count['count']}")
    
    # Sample search
    sample_query = {
        "query": {
            "match_all": {}
        },
        "size": 3
    }
    results = opensearch_client.search(index=OPENSEARCH_INDEX, body=sample_query)
    print(f"\nSample documents:")
    for hit in results['hits']['hits']:
        print(f"- {hit['_source']['metadata']['source']}: {hit['_source']['content'][:100]}...")
    ```
    - Run the cell

11. **Save Notebook**
    - Click "File" → "Save Notebook As..."
    - Name: "document-processing-pipeline.ipynb"
    - Save in your rag-production folder


### Phase 7: RAG Orchestration Implementation

This phase implements the core RAG logic for query processing and response generation.

#### Component 7.1: Create RAG Query Notebook

**Purpose**: Build the RAG query pipeline that retrieves context and generates responses.

**AWS GUI Step-by-Step Instructions**:

1. **Create New Notebook in Studio**
   - File → New → Notebook
   - Kernel: Python 3 (Data Science 3.0)

2. **Setup and Configuration**
   - First cell:
   ```python
   import boto3
   import json
   from opensearchpy import OpenSearch, RequestsHttpConnection
   
   # Initialize clients
   bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
   sagemaker_runtime = boto3.client('sagemaker-runtime')
   
   # Configuration
   EMBEDDING_ENDPOINT = 'rag-embedding-endpoint'
   OPENSEARCH_ENDPOINT = 'vpc-rag-vector-db-abc123.us-east-1.es.amazonaws.com'
   OPENSEARCH_INDEX = 'rag-documents-index'
   OPENSEARCH_USERNAME = 'admin'
   OPENSEARCH_PASSWORD = '[your-password]'
   BEDROCK_MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'
   
   # Connect to OpenSearch
   opensearch_client = OpenSearch(
       hosts=[{'host': OPENSEARCH_ENDPOINT, 'port': 443}],
       http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
       use_ssl=True,
       verify_certs=True,
       connection_class=RequestsHttpConnection
   )
   ```

3. **Create Query Embedding Function**
   - New cell:
   ```python
   def get_query_embedding(query_text):
       """Convert query to embedding vector"""
       payload = {"text_inputs": [query_text]}
       response = sagemaker_runtime.invoke_endpoint(
           EndpointName=EMBEDDING_ENDPOINT,
           ContentType='application/json',
           Body=json.dumps(payload)
       )
       result = json.loads(response['Body'].read())
       return result['embedding'][0]
   ```

4. **Create Vector Search Function**
   - New cell:
   ```python
   def search_similar_documents(query_embedding, top_k=5):
       """Search for similar documents using k-NN"""
       search_query = {
           "size": top_k,
           "query": {
               "knn": {
                   "embedding": {
                       "vector": query_embedding,
                       "k": top_k
                   }
               }
           },
           "_source": ["content", "metadata"]
       }
       
       results = opensearch_client.search(
           index=OPENSEARCH_INDEX,
           body=search_query
       )
       
       documents = []
       for hit in results['hits']['hits']:
           documents.append({
               'content': hit['_source']['content'],
               'source': hit['_source']['metadata']['source'],
               'score': hit['_score']
           })
       
       return documents
   ```

5. **Create Prompt Construction Function**
   - New cell:
   ```python
   def construct_rag_prompt(query, retrieved_docs):
       """Build prompt with context and query"""
       context = "\n\n".join([
           f"Document {i+1} (Source: {doc['source']}):\n{doc['content']}"
           for i, doc in enumerate(retrieved_docs)
       ])
       
       prompt = f"""You are a helpful AI assistant. Answer the question based only on the provided context. 
   If the answer is not in the context, say "I don't have enough information to answer that question."
   
   Context:
   {context}
   
   Question: {query}
   
   Answer:"""
       
       return prompt
   ```

6. **Create Bedrock Generation Function**
   - New cell:
   ```python
   def generate_response(prompt):
       """Generate response using Bedrock"""
       body = json.dumps({
           "anthropic_version": "bedrock-2023-05-31",
           "max_tokens": 2048,
           "temperature": 0.7,
           "top_p": 0.9,
           "messages": [
               {
                   "role": "user",
                   "content": prompt
               }
           ]
       })
       
       response = bedrock_runtime.invoke_model(
           modelId=BEDROCK_MODEL_ID,
           body=body
       )
       
       response_body = json.loads(response['body'].read())
       return response_body['content'][0]['text']
   ```

7. **Create Complete RAG Pipeline**
   - New cell:
   ```python
   def rag_query(user_question, top_k=5, verbose=True):
       """Complete RAG pipeline"""
       if verbose:
           print(f"Question: {user_question}\n")
           print("Step 1: Converting query to embedding...")
       
       # Get query embedding
       query_embedding = get_query_embedding(user_question)
       
       if verbose:
           print("Step 2: Searching for relevant documents...")
       
       # Search similar documents
       retrieved_docs = search_similar_documents(query_embedding, top_k)
       
       if verbose:
           print(f"Step 3: Found {len(retrieved_docs)} relevant documents")
           for i, doc in enumerate(retrieved_docs):
               print(f"  - Document {i+1}: {doc['source']} (score: {doc['score']:.4f})")
           print("\nStep 4: Generating response...")
       
       # Construct prompt
       prompt = construct_rag_prompt(user_question, retrieved_docs)
       
       # Generate response
       answer = generate_response(prompt)
       
       if verbose:
           print("\n" + "="*80)
           print("ANSWER:")
           print("="*80)
           print(answer)
           print("="*80)
           print("\nSOURCES:")
           for doc in retrieved_docs:
               print(f"  - {doc['source']}")
       
       return {
           'answer': answer,
           'sources': [doc['source'] for doc in retrieved_docs],
           'retrieved_docs': retrieved_docs
       }
   ```

8. **Test RAG System**
   - New cell:
   ```python
   # Test with sample questions
   test_questions = [
       "What is Amazon SageMaker?",
       "How do I deploy a model?",
       "What are the benefits of using this service?"
   ]
   
   for question in test_questions:
       result = rag_query(question, top_k=3)
       print("\n" + "="*100 + "\n")
   ```

9. **Save Notebook**
   - File → Save Notebook As...
   - Name: "rag-query-pipeline.ipynb"


### Phase 8: Bedrock Guardrails Implementation

This phase adds security controls to filter harmful content and ensure responsible AI usage.

#### Component 8.1: Create Bedrock Guardrail

**Purpose**: Configure content filters and safety controls for the RAG system.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to Bedrock Guardrails**
   - In AWS Console, go to Amazon Bedrock
   - In left sidebar, click "Guardrails" under "Safeguards"

2. **Create New Guardrail**
   - Click "Create guardrail" button
   - **Guardrail name**: "rag-production-guardrail"
   - **Description**: "Content filtering and safety controls for RAG application"

3. **Configure Content Filters**
   - Scroll to "Content filters" section
   - Click "Configure content filters"
   - Set filter strengths for each category:
     - **Hate**: HIGH
     - **Insults**: MEDIUM
     - **Sexual**: HIGH
     - **Violence**: HIGH
     - **Misconduct**: MEDIUM
     - **Prompt attacks**: HIGH (prevents prompt injection)
   - Toggle "Filter input" and "Filter output" for all categories

4. **Configure Denied Topics** (Optional)
   - Scroll to "Denied topics" section
   - Click "Add denied topic"
   - **Topic name**: "Financial Advice"
   - **Definition**: "Requests for specific financial, investment, or trading advice"
   - **Example phrases**: Add 3-5 examples:
     - "Should I invest in stocks?"
     - "What cryptocurrency should I buy?"
     - "Give me financial advice"
   - Click "Add topic"
   - Repeat for other sensitive topics relevant to your use case

5. **Configure Word Filters** (Optional)
   - Scroll to "Word filters" section
   - Click "Add word filter"
   - Add profanity or inappropriate terms to block
   - Select "Filter input" and "Filter output"

6. **Configure Sensitive Information Filters** (Optional)
   - Scroll to "Sensitive information filters"
   - Select PII types to redact:
     - ✓ Email addresses
     - ✓ Phone numbers
     - ✓ Credit card numbers
     - ✓ Social Security numbers
   - Choose action: "Block" or "Anonymize"

7. **Configure Contextual Grounding** (Important for RAG)
   - Scroll to "Contextual grounding" section
   - Enable "Grounding check"
   - **Grounding threshold**: 0.75 (blocks responses not grounded in context)
   - **Relevance threshold**: 0.70 (blocks irrelevant responses)
   - This prevents hallucinations in RAG responses

8. **Review and Create**
   - Scroll to bottom
   - Review all configurations
   - Click "Create guardrail"
   - Wait for creation (1-2 minutes)

9. **Create Guardrail Version**
   - After creation, click "Create version" button
   - **Version description**: "Initial production version"
   - Click "Create version"
   - Note the version number (e.g., "1")

10. **Note Guardrail Details** (save for later)
    - Guardrail ID (e.g., abc123def456)
    - Guardrail ARN
    - Version number


#### Component 8.2: Integrate Guardrails into RAG Pipeline

**Purpose**: Update the RAG query function to use Bedrock Guardrails.

**AWS GUI Step-by-Step Instructions**:

1. **Return to SageMaker Studio**
   - Open your "rag-query-pipeline.ipynb" notebook

2. **Update Configuration**
   - Add to configuration cell:
   ```python
   GUARDRAIL_ID = 'abc123def456'  # Your guardrail ID
   GUARDRAIL_VERSION = '1'  # Your version number
   ```

3. **Update Bedrock Generation Function**
   - Replace the generate_response function:
   ```python
   def generate_response_with_guardrails(prompt):
       """Generate response using Bedrock with Guardrails"""
       body = json.dumps({
           "anthropic_version": "bedrock-2023-05-31",
           "max_tokens": 2048,
           "temperature": 0.7,
           "top_p": 0.9,
           "messages": [
               {
                   "role": "user",
                   "content": prompt
               }
           ]
       })
       
       try:
           response = bedrock_runtime.invoke_model(
               modelId=BEDROCK_MODEL_ID,
               body=body,
               guardrailIdentifier=GUARDRAIL_ID,
               guardrailVersion=GUARDRAIL_VERSION
           )
           
           response_body = json.loads(response['body'].read())
           
           # Check if response was blocked by guardrails
           if 'amazon-bedrock-guardrailAction' in response['ResponseMetadata']['HTTPHeaders']:
               action = response['ResponseMetadata']['HTTPHeaders']['amazon-bedrock-guardrailAction']
               if action == 'GUARDRAIL_INTERVENED':
                   return "I apologize, but I cannot provide a response to that query due to content policy restrictions."
           
           return response_body['content'][0]['text']
           
       except Exception as e:
           if 'ValidationException' in str(e):
               return "I apologize, but your query was blocked by our content filters."
           raise e
   ```

4. **Update RAG Pipeline**
   - Modify the rag_query function to use the new function:
   ```python
   # Replace this line:
   # answer = generate_response(prompt)
   # With:
   answer = generate_response_with_guardrails(prompt)
   ```

5. **Test Guardrails**
   - New cell to test blocked content:
   ```python
   # Test with potentially harmful query
   test_query = "How do I hack into a system?"
   result = rag_query(test_query, top_k=3)
   
   # Should return blocked message
   print(result['answer'])
   ```

6. **Test Grounding Check**
   - New cell:
   ```python
   # Test with query that might cause hallucination
   test_query = "What is the capital of Mars?"
   result = rag_query(test_query, top_k=3)
   
   # Should indicate lack of information
   print(result['answer'])
   ```

7. **Save Updated Notebook**
   - File → Save Notebook

### Phase 9: Monitoring and Logging Setup

This phase implements comprehensive monitoring for the RAG system.

#### Component 9.1: Create CloudWatch Dashboard

**Purpose**: Set up centralized monitoring for all RAG components.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to CloudWatch**
   - In AWS Console, search for "CloudWatch"
   - Click on "CloudWatch"

2. **Create Dashboard**
   - In left sidebar, click "Dashboards"
   - Click "Create dashboard" button
   - **Dashboard name**: "RAG-Production-Dashboard"
   - Click "Create dashboard"

3. **Add SageMaker Endpoint Metrics**
   - Click "Add widget" button
   - Select "Line" chart type
   - Click "Next"
   - **Data source**: Metrics
   - In metrics browser:
     - Namespace: AWS/SageMaker
     - Metric name: Select "ModelLatency"
     - Dimensions: EndpointName = rag-embedding-endpoint
   - Click "Add metric"
   - Also add:
     - Invocations
     - InvocationsPerInstance
     - ModelSetupTime
   - Click "Create widget"

4. **Add OpenSearch Metrics**
   - Click "Add widget" (+ icon)
   - Select "Line" chart
   - Namespace: AWS/ES
   - Select metrics:
     - ClusterStatus.green
     - SearchRate
     - SearchLatency
     - IndexingRate
   - Dimensions: DomainName = rag-vector-db
   - Click "Create widget"

5. **Add Bedrock Metrics**
   - Click "Add widget"
   - Select "Number" widget type
   - Namespace: AWS/Bedrock
   - Select metrics:
     - Invocations
     - InvocationLatency
     - InvocationClientErrors
     - InvocationServerErrors
   - Click "Create widget"

6. **Add Cost Metrics**
   - Click "Add widget"
   - Select "Number" widget
   - Create custom metric for estimated costs
   - Or use AWS Cost Explorer data

7. **Arrange Dashboard**
   - Drag and resize widgets for optimal layout
   - Click "Save dashboard"


#### Component 9.2: Configure CloudWatch Alarms

**Purpose**: Set up alerts for system issues and performance degradation.

**AWS GUI Step-by-Step Instructions**:

1. **Create Alarm for Endpoint Latency**
   - In CloudWatch console, click "Alarms" in left sidebar
   - Click "Create alarm" button
   - Click "Select metric"
   - Navigate to: SageMaker → Endpoint Metrics
   - Select "ModelLatency" for rag-embedding-endpoint
   - Click "Select metric"

2. **Configure Alarm Conditions**
   - **Statistic**: Average
   - **Period**: 5 minutes
   - **Threshold type**: Static
   - **Condition**: Greater than 5000 (5 seconds)
   - Click "Next"

3. **Configure Alarm Actions**
   - **Alarm state trigger**: In alarm
   - **Send notification to**: Create new SNS topic
   - **Topic name**: "rag-alerts"
   - **Email endpoints**: Enter your email address
   - Click "Create topic"
   - Click "Next"

4. **Name and Describe Alarm**
   - **Alarm name**: "RAG-Embedding-High-Latency"
   - **Description**: "Alert when embedding endpoint latency exceeds 5 seconds"
   - Click "Next"
   - Review and click "Create alarm"

5. **Confirm SNS Subscription**
   - Check your email
   - Click confirmation link in AWS SNS email
   - Return to CloudWatch console

6. **Create Alarm for Endpoint Errors**
   - Click "Create alarm" again
   - Select metric: SageMaker → ModelInvocationErrors
   - Endpoint: rag-embedding-endpoint
   - **Condition**: Greater than 10 (errors in 5 minutes)
   - **SNS topic**: Select existing "rag-alerts"
   - **Alarm name**: "RAG-Embedding-High-Errors"
   - Create alarm

7. **Create Alarm for OpenSearch Health**
   - Create new alarm
   - Metric: AWS/ES → ClusterStatus.red
   - Domain: rag-vector-db
   - **Condition**: Greater than or equal to 1
   - **SNS topic**: rag-alerts
   - **Alarm name**: "RAG-OpenSearch-Cluster-Red"
   - Create alarm

8. **Create Alarm for Bedrock Throttling**
   - Create new alarm
   - Metric: AWS/Bedrock → InvocationThrottles
   - **Condition**: Greater than 5
   - **SNS topic**: rag-alerts
   - **Alarm name**: "RAG-Bedrock-Throttling"
   - Create alarm

9. **Verify Alarms**
   - Go to "Alarms" page
   - You should see all 4 alarms with "OK" status
   - Click on each to verify configuration

#### Component 9.3: Enable Detailed Logging

**Purpose**: Capture detailed logs for debugging and audit trails.

**AWS GUI Step-by-Step Instructions**:

1. **Enable SageMaker Endpoint Logging**
   - Go to SageMaker console
   - Click "Endpoints" in left sidebar
   - Click on "rag-embedding-endpoint"
   - Click "Update endpoint" button
   - Scroll to "Data capture"
   - **Enable data capture**: Check the box
   - **Sampling percentage**: 100 (for full logging, reduce for production)
   - **Destination S3 URI**: s3://rag-logs-[account-id]-[region]/sagemaker-data-capture/
   - Click "Update endpoint"

2. **Configure CloudWatch Logs for Lambda** (for future API)
   - We'll configure this when creating Lambda functions
   - Lambda automatically logs to CloudWatch Logs

3. **Enable CloudTrail for Audit Logs**
   - Search for "CloudTrail" in AWS Console
   - Click "Trails" in left sidebar
   - Click "Create trail"
   - **Trail name**: "rag-audit-trail"
   - **Storage location**: Create new S3 bucket
   - **Bucket name**: rag-cloudtrail-[account-id]-[region]
   - **Log file SSE-KMS encryption**: Enable
   - Click "Next"

4. **Choose Log Events**
   - **Management events**: Enable
   - **Data events**: Enable
   - Add data event for S3:
     - Select "S3"
     - Select "All current and future S3 buckets"
     - Or select specific RAG buckets
   - Click "Next"
   - Review and click "Create trail"

5. **View Logs in CloudWatch**
   - Go to CloudWatch console
   - Click "Log groups" in left sidebar
   - You should see log groups for:
     - /aws/sagemaker/Endpoints/rag-embedding-endpoint
     - /aws/opensearch/domains/rag-vector-db (if enabled)
   - Click on any log group to view logs

### Phase 10: API Gateway and Lambda Integration

This phase creates REST API endpoints for programmatic access to the RAG system.

#### Component 10.1: Create Lambda Function for RAG

**Purpose**: Package the RAG logic into a serverless function.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to Lambda Service**
   - In AWS Console, search for "Lambda"
   - Click on "AWS Lambda"

2. **Create Function**
   - Click "Create function" button
   - Select "Author from scratch"
   - **Function name**: "rag-query-function"
   - **Runtime**: Python 3.11
   - **Architecture**: x86_64

3. **Configure Permissions**
   - **Execution role**: Use an existing role
   - Select "RAGLambdaExecutionRole" (created earlier)
   - Click "Create function"

4. **Configure Function Settings**
   - In the function page, click "Configuration" tab
   - Click "General configuration" → "Edit"
   - **Memory**: 1024 MB
   - **Timeout**: 5 minutes (300 seconds)
   - **Ephemeral storage**: 512 MB
   - Click "Save"

5. **Configure VPC** (to access OpenSearch)
   - Still in Configuration tab, click "VPC"
   - Click "Edit"
   - **VPC**: Select "rag-production-vpc"
   - **Subnets**: Select both private subnets
   - **Security groups**: Select "rag-sagemaker-sg"
   - Click "Save"
   - Wait 2-3 minutes for VPC configuration


6. **Add Environment Variables**
   - In Configuration tab, click "Environment variables"
   - Click "Edit"
   - Add these variables:
     - EMBEDDING_ENDPOINT = rag-embedding-endpoint
     - OPENSEARCH_ENDPOINT = vpc-rag-vector-db-abc123.us-east-1.es.amazonaws.com
     - OPENSEARCH_INDEX = rag-documents-index
     - OPENSEARCH_USERNAME = admin
     - OPENSEARCH_PASSWORD = [your-password]
     - BEDROCK_MODEL_ID = anthropic.claude-3-sonnet-20240229-v1:0
     - GUARDRAIL_ID = [your-guardrail-id]
     - GUARDRAIL_VERSION = 1
   - Click "Save"

7. **Add Lambda Layer for Dependencies**
   - Click "Code" tab
   - Scroll down to "Layers" section
   - Click "Add a layer"
   - Select "AWS layers"
   - Choose "AWSSDKPandas-Python311" (includes boto3, opensearch-py)
   - Click "Add"

8. **Write Lambda Function Code**
   - In the Code tab, replace the default code with:
   ```python
   import json
   import boto3
   import os
   from opensearchpy import OpenSearch, RequestsHttpConnection
   
   # Initialize clients
   sagemaker_runtime = boto3.client('sagemaker-runtime')
   bedrock_runtime = boto3.client('bedrock-runtime')
   
   # Get configuration from environment
   EMBEDDING_ENDPOINT = os.environ['EMBEDDING_ENDPOINT']
   OPENSEARCH_ENDPOINT = os.environ['OPENSEARCH_ENDPOINT']
   OPENSEARCH_INDEX = os.environ['OPENSEARCH_INDEX']
   OPENSEARCH_USERNAME = os.environ['OPENSEARCH_USERNAME']
   OPENSEARCH_PASSWORD = os.environ['OPENSEARCH_PASSWORD']
   BEDROCK_MODEL_ID = os.environ['BEDROCK_MODEL_ID']
   GUARDRAIL_ID = os.environ['GUARDRAIL_ID']
   GUARDRAIL_VERSION = os.environ['GUARDRAIL_VERSION']
   
   # Initialize OpenSearch client
   opensearch_client = OpenSearch(
       hosts=[{'host': OPENSEARCH_ENDPOINT, 'port': 443}],
       http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
       use_ssl=True,
       verify_certs=True,
       connection_class=RequestsHttpConnection
   )
   
   def get_query_embedding(query_text):
       payload = {"text_inputs": [query_text]}
       response = sagemaker_runtime.invoke_endpoint(
           EndpointName=EMBEDDING_ENDPOINT,
           ContentType='application/json',
           Body=json.dumps(payload)
       )
       result = json.loads(response['Body'].read())
       return result['embedding'][0]
   
   def search_similar_documents(query_embedding, top_k=5):
       search_query = {
           "size": top_k,
           "query": {
               "knn": {
                   "embedding": {
                       "vector": query_embedding,
                       "k": top_k
                   }
               }
           },
           "_source": ["content", "metadata"]
       }
       
       results = opensearch_client.search(
           index=OPENSEARCH_INDEX,
           body=search_query
       )
       
       documents = []
       for hit in results['hits']['hits']:
           documents.append({
               'content': hit['_source']['content'],
               'source': hit['_source']['metadata']['source'],
               'score': hit['_score']
           })
       
       return documents
   
   def construct_rag_prompt(query, retrieved_docs):
       context = "\n\n".join([
           f"Document {i+1}:\n{doc['content']}"
           for i, doc in enumerate(retrieved_docs)
       ])
       
       prompt = f"""Answer the question based only on the provided context.
   
   Context:
   {context}
   
   Question: {query}
   
   Answer:"""
       
       return prompt
   
   def generate_response(prompt):
       body = json.dumps({
           "anthropic_version": "bedrock-2023-05-31",
           "max_tokens": 2048,
           "temperature": 0.7,
           "messages": [{"role": "user", "content": prompt}]
       })
       
       response = bedrock_runtime.invoke_model(
           modelId=BEDROCK_MODEL_ID,
           body=body,
           guardrailIdentifier=GUARDRAIL_ID,
           guardrailVersion=GUARDRAIL_VERSION
       )
       
       response_body = json.loads(response['body'].read())
       return response_body['content'][0]['text']
   
   def lambda_handler(event, context):
       try:
           # Parse request
           body = json.loads(event['body']) if isinstance(event.get('body'), str) else event
           query = body.get('query', '')
           top_k = body.get('top_k', 5)
           
           if not query:
               return {
                   'statusCode': 400,
                   'body': json.dumps({'error': 'Query parameter is required'})
               }
           
           # Execute RAG pipeline
           query_embedding = get_query_embedding(query)
           retrieved_docs = search_similar_documents(query_embedding, top_k)
           prompt = construct_rag_prompt(query, retrieved_docs)
           answer = generate_response(prompt)
           
           # Return response
           return {
               'statusCode': 200,
               'headers': {
                   'Content-Type': 'application/json',
                   'Access-Control-Allow-Origin': '*'
               },
               'body': json.dumps({
                   'answer': answer,
                   'sources': [doc['source'] for doc in retrieved_docs],
                   'num_sources': len(retrieved_docs)
               })
           }
           
       except Exception as e:
           print(f"Error: {str(e)}")
           return {
               'statusCode': 500,
               'body': json.dumps({'error': str(e)})
           }
   ```
   - Click "Deploy" button to save

9. **Test Lambda Function**
   - Click "Test" tab
   - Click "Create new event"
   - **Event name**: "test-query"
   - **Event JSON**:
   ```json
   {
     "body": "{\"query\": \"What is Amazon SageMaker?\", \"top_k\": 3}"
   }
   ```
   - Click "Save"
   - Click "Test" button
   - Review execution results (should see answer in response)

10. **Note Lambda Details** (save for later)
    - Function ARN
    - Function name


#### Component 10.2: Create API Gateway

**Purpose**: Expose the Lambda function as a REST API endpoint.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to API Gateway**
   - In AWS Console, search for "API Gateway"
   - Click on "API Gateway"

2. **Create REST API**
   - Click "Create API" button
   - Find "REST API" (not Private)
   - Click "Build" button
   - Select "New API"
   - **API name**: "RAG-Production-API"
   - **Description**: "REST API for RAG query system"
   - **Endpoint Type**: Regional
   - Click "Create API"

3. **Create Resource**
   - Click "Actions" dropdown → "Create Resource"
   - **Resource Name**: "query"
   - **Resource Path**: /query
   - **Enable API Gateway CORS**: Check this box
   - Click "Create Resource"

4. **Create POST Method**
   - With "/query" resource selected, click "Actions" → "Create Method"
   - Select "POST" from dropdown
   - Click the checkmark
   - **Integration type**: Lambda Function
   - **Use Lambda Proxy integration**: Check this box
   - **Lambda Region**: Select your region
   - **Lambda Function**: Type "rag-query-function"
   - Click "Save"
   - Click "OK" to grant API Gateway permission to invoke Lambda

5. **Enable CORS**
   - With "/query" selected, click "Actions" → "Enable CORS"
   - Keep default settings
   - Click "Enable CORS and replace existing CORS headers"
   - Click "Yes, replace existing values"

6. **Create API Key** (for authentication)
   - In left sidebar, click "API Keys"
   - Click "Actions" → "Create API Key"
   - **Name**: "rag-client-key"
   - **Auto Generate**: Select this
   - Click "Save"
   - Click "Show" to reveal the API key
   - **Copy and save this key securely**

7. **Create Usage Plan**
   - In left sidebar, click "Usage Plans"
   - Click "Create" button
   - **Name**: "rag-basic-plan"
   - **Description**: "Basic usage plan for RAG API"
   - **Enable throttling**: Check
     - **Rate**: 100 requests per second
     - **Burst**: 200 requests
   - **Enable quota**: Check
     - **Quota**: 10000 requests per month
   - Click "Next"

8. **Associate API Stage**
   - Click "Add API Stage"
   - **API**: Select "RAG-Production-API"
   - **Stage**: We'll create this next, skip for now
   - Click "Done" (we'll come back)

9. **Deploy API**
   - In left sidebar, click "Resources"
   - Click "Actions" → "Deploy API"
   - **Deployment stage**: [New Stage]
   - **Stage name**: "prod"
   - **Stage description**: "Production stage"
   - Click "Deploy"

10. **Complete Usage Plan Association**
    - Go back to "Usage Plans"
    - Click on "rag-basic-plan"
    - Click "API Stages" tab
    - Click "Add API Stage"
    - **API**: RAG-Production-API
    - **Stage**: prod
    - Click checkmark
    - Click "Associated API Keys" tab
    - Click "Add API Key to Usage Plan"
    - Select "rag-client-key"
    - Click checkmark

11. **Configure Method Request** (require API key)
    - Go to "Resources"
    - Click on "POST" method under /query
    - Click "Method Request"
    - **API Key Required**: Change to "true"
    - Click checkmark
    - Click "Actions" → "Deploy API"
    - Select "prod" stage
    - Click "Deploy"

12. **Get API Endpoint URL**
    - Click "Stages" in left sidebar
    - Click "prod"
    - You'll see "Invoke URL" at the top
    - Copy this URL (e.g., https://abc123.execute-api.us-east-1.amazonaws.com/prod)
    - Your full endpoint: https://abc123.execute-api.us-east-1.amazonaws.com/prod/query

13. **Test API**
    - Use curl or Postman to test:
    ```bash
    curl -X POST https://abc123.execute-api.us-east-1.amazonaws.com/prod/query \
      -H "Content-Type: application/json" \
      -H "x-api-key: YOUR_API_KEY" \
      -d '{"query": "What is Amazon SageMaker?", "top_k": 3}'
    ```

14. **Note API Details** (save for later)
    - API endpoint URL
    - API key
    - Stage name

### Phase 11: Performance Optimization

This phase optimizes the RAG system for better performance and cost efficiency.

#### Component 11.1: Configure Auto-Scaling for SageMaker Endpoint

**Purpose**: Automatically scale endpoint capacity based on traffic.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to SageMaker Console**
   - Go to SageMaker service
   - Click "Endpoints" in left sidebar
   - Click on "rag-embedding-endpoint"

2. **Configure Auto-Scaling**
   - Click "Configure auto scaling" button
   - Or click "Update endpoint" → scroll to "Endpoint runtime settings"

3. **Set Scaling Policy**
   - **Variant name**: Select the variant (usually "AllTraffic")
   - **Minimum instance count**: 1
   - **Maximum instance count**: 5
   - Click "Configure variant auto scaling"

4. **Create Scaling Policy**
   - **Policy name**: "rag-embedding-scaling-policy"
   - **Target metric**: SageMakerVariantInvocationsPerInstance
   - **Target value**: 1000 (invocations per instance)
   - **Scale-in cool down**: 300 seconds
   - **Scale-out cool down**: 60 seconds
   - Click "Save"

5. **Verify Auto-Scaling**
   - Go to "Application Auto Scaling" in AWS Console
   - You should see the scaling policy listed
   - Monitor scaling activities in CloudWatch


#### Component 11.2: Implement Caching Strategy

**Purpose**: Reduce redundant API calls and improve response times.

**AWS GUI Step-by-Step Instructions**:

1. **Create ElastiCache Redis Cluster** (Optional but Recommended)
   - In AWS Console, search for "ElastiCache"
   - Click "Create" button
   - Select "Redis"
   - **Cluster mode**: Disabled (for simplicity)
   - **Name**: "rag-cache"
   - **Node type**: cache.t3.micro (for development) or cache.r6g.large (production)
   - **Number of replicas**: 1 (for high availability)
   - **Subnet group**: Create new
     - Select "rag-production-vpc"
     - Select private subnets
   - **Security group**: Select "rag-sagemaker-sg"
   - Click "Create"
   - Wait 10-15 minutes for creation

2. **Update Lambda Function for Caching**
   - Go to Lambda console
   - Open "rag-query-function"
   - Add environment variable:
     - REDIS_ENDPOINT = [your-redis-endpoint]:6379
   - Update code to include caching logic (add at top):
   ```python
   import hashlib
   import redis
   
   # Initialize Redis client
   redis_client = redis.Redis(
       host=os.environ.get('REDIS_ENDPOINT', '').split(':')[0],
       port=6379,
       decode_responses=True
   )
   
   def get_cache_key(query):
       return f"rag:{hashlib.md5(query.encode()).hexdigest()}"
   
   # In lambda_handler, add caching:
   cache_key = get_cache_key(query)
   cached_result = redis_client.get(cache_key)
   
   if cached_result:
       return {
           'statusCode': 200,
           'body': cached_result,
           'headers': {'X-Cache': 'HIT'}
       }
   
   # ... existing RAG logic ...
   
   # Before returning, cache the result:
   redis_client.setex(cache_key, 3600, json.dumps(response_body))  # 1 hour TTL
   ```

3. **Enable API Gateway Caching**
   - Go to API Gateway console
   - Select "RAG-Production-API"
   - Click "Stages" → "prod"
   - Click "Settings" tab
   - **Enable API cache**: Check
   - **Cache capacity**: 0.5 GB (adjust based on needs)
   - **Cache time-to-live (TTL)**: 300 seconds (5 minutes)
   - Click "Save Changes"
   - Click "Actions" → "Deploy API" → select "prod"

#### Component 11.3: Optimize OpenSearch Performance

**Purpose**: Tune OpenSearch for better vector search performance.

**AWS GUI Step-by-Step Instructions**:

1. **Access OpenSearch Dashboards**
   - Go to OpenSearch Service console
   - Click on "rag-vector-db"
   - Click OpenSearch Dashboards URL

2. **Optimize Index Settings**
   - In Dev Tools, run:
   ```json
   PUT /rag-documents-index/_settings
   {
     "index": {
       "refresh_interval": "30s",
       "number_of_replicas": 1,
       "translog.durability": "async",
       "translog.sync_interval": "30s"
     }
   }
   ```

3. **Create Index Alias** (for zero-downtime updates)
   ```json
   POST /_aliases
   {
     "actions": [
       {
         "add": {
           "index": "rag-documents-index",
           "alias": "rag-documents"
         }
       }
     ]
   }
   ```

4. **Monitor Query Performance**
   - In Dashboards, go to "Dev Tools"
   - Run:
   ```json
   GET /rag-documents-index/_search
   {
     "profile": true,
     "query": {
       "match_all": {}
     }
   }
   ```
   - Review the "profile" section for slow operations

### Phase 12: Backup and Disaster Recovery

This phase implements backup strategies and recovery procedures.

#### Component 12.1: Configure Automated Backups

**Purpose**: Ensure data can be recovered in case of failures.

**AWS GUI Step-by-Step Instructions**:

1. **Enable S3 Versioning** (Already done in Phase 1)
   - Verify in S3 console that versioning is enabled
   - Go to each bucket → Properties → Versioning

2. **Configure S3 Lifecycle Policies**
   - In S3 console, click on "rag-documents-..." bucket
   - Click "Management" tab
   - Click "Create lifecycle rule"
   - **Rule name**: "archive-old-versions"
   - **Rule scope**: Apply to all objects
   - **Lifecycle rule actions**:
     - ✓ Transition noncurrent versions between storage classes
     - ✓ Permanently delete noncurrent versions
   - **Transition noncurrent versions**:
     - After 30 days → Glacier Flexible Retrieval
   - **Permanently delete noncurrent versions**:
     - After 90 days
   - Click "Create rule"

3. **Configure OpenSearch Snapshots**
   - Go to OpenSearch Service console
   - Click on "rag-vector-db"
   - Note: Automated snapshots are already enabled (configured during creation)
   - Manual snapshot creation:
     - In OpenSearch Dashboards Dev Tools:
     ```json
     PUT /_snapshot/manual-snapshots/snapshot-1
     {
       "indices": "rag-documents-index",
       "ignore_unavailable": true,
       "include_global_state": false
     }
     ```

4. **Create Backup Lambda Function** (Optional)
   - Create Lambda to periodically backup configurations
   - Store in S3 for disaster recovery

5. **Document Recovery Procedures**
   - Create a document with step-by-step recovery instructions
   - Store in S3 and version control


#### Component 12.2: Test Disaster Recovery

**Purpose**: Validate that recovery procedures work correctly.

**AWS GUI Step-by-Step Instructions**:

1. **Test OpenSearch Snapshot Restore**
   - In OpenSearch Dashboards Dev Tools:
   ```json
   # List available snapshots
   GET /_snapshot/cs-automated/_all
   
   # Restore from snapshot (to new index)
   POST /_snapshot/cs-automated/snapshot-name/_restore
   {
     "indices": "rag-documents-index",
     "rename_pattern": "(.+)",
     "rename_replacement": "restored-$1"
   }
   ```

2. **Test S3 Version Recovery**
   - Go to S3 console
   - Navigate to a file in rag-documents bucket
   - Click "Versions" toggle
   - Select an older version
   - Click "Download" or "Restore"

3. **Document Test Results**
   - Record recovery time objectives (RTO)
   - Record recovery point objectives (RPO)
   - Update disaster recovery documentation

### Phase 13: Security Hardening

This phase implements additional security measures for production readiness.

#### Component 13.1: Implement Secrets Management

**Purpose**: Securely store and manage sensitive credentials.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to Secrets Manager**
   - In AWS Console, search for "Secrets Manager"
   - Click "Secrets Manager"

2. **Store OpenSearch Credentials**
   - Click "Store a new secret"
   - **Secret type**: Other type of secret
   - **Key/value pairs**:
     - Key: username, Value: admin
     - Key: password, Value: [your-opensearch-password]
   - **Encryption key**: Use default (aws/secretsmanager)
   - Click "Next"
   - **Secret name**: "rag/opensearch/credentials"
   - **Description**: "OpenSearch admin credentials for RAG system"
   - Click "Next"
   - **Automatic rotation**: Disable (or configure if needed)
   - Click "Next"
   - Review and click "Store"

3. **Store API Keys**
   - Create another secret
   - **Key/value pairs**:
     - Key: api_key, Value: [your-api-gateway-key]
   - **Secret name**: "rag/api/keys"
   - Store secret

4. **Update Lambda to Use Secrets Manager**
   - Go to Lambda console
   - Open "rag-query-function"
   - Update code to retrieve secrets:
   ```python
   import boto3
   import json
   
   secrets_client = boto3.client('secretsmanager')
   
   def get_secret(secret_name):
       response = secrets_client.get_secret_value(SecretId=secret_name)
       return json.loads(response['SecretString'])
   
   # At initialization:
   opensearch_creds = get_secret('rag/opensearch/credentials')
   OPENSEARCH_USERNAME = opensearch_creds['username']
   OPENSEARCH_PASSWORD = opensearch_creds['password']
   ```
   - Remove credentials from environment variables
   - Deploy updated function

5. **Update IAM Role**
   - Go to IAM console
   - Open "RAGLambdaExecutionRole"
   - Add inline policy:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "secretsmanager:GetSecretValue"
         ],
         "Resource": [
           "arn:aws:secretsmanager:*:*:secret:rag/*"
         ]
       }
     ]
   }
   ```
   - Name: "SecretsManagerAccess"
   - Create policy

#### Component 13.2: Enable AWS WAF

**Purpose**: Protect API Gateway from common web exploits.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to AWS WAF**
   - In AWS Console, search for "WAF"
   - Click "AWS WAF & Shield"

2. **Create Web ACL**
   - Click "Create web ACL"
   - **Name**: "rag-api-protection"
   - **Resource type**: Regional resources (API Gateway)
   - **Region**: Select your region
   - Click "Next"

3. **Add Managed Rule Groups**
   - Click "Add rules" → "Add managed rule groups"
   - Expand "AWS managed rule groups"
   - Select these rule groups:
     - ✓ Core rule set
     - ✓ Known bad inputs
     - ✓ SQL database
     - ✓ Linux operating system
   - Click "Add rules"
   - Click "Next"

4. **Set Rule Priority**
   - Keep default priority order
   - Click "Next"

5. **Configure Metrics**
   - **CloudWatch metrics**: Enable
   - **Sampled requests**: Enable
   - Click "Next"

6. **Review and Create**
   - Review all settings
   - Click "Create web ACL"

7. **Associate with API Gateway**
   - After creation, click "Associated AWS resources"
   - Click "Add AWS resources"
   - **Resource type**: API Gateway
   - Select "RAG-Production-API" in "prod" stage
   - Click "Add"

8. **Test WAF Rules**
   - Try sending malicious requests to test blocking
   - Monitor in WAF console under "Overview"

### Phase 14: Cost Optimization

This phase implements strategies to reduce operational costs.

#### Component 14.1: Implement Cost Monitoring

**Purpose**: Track and optimize spending on RAG infrastructure.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to Cost Explorer**
   - In AWS Console, search for "Cost Explorer"
   - Click "AWS Cost Explorer"
   - Click "Enable Cost Explorer" if not already enabled

2. **Create Cost Report**
   - Click "Reports" in left sidebar
   - Click "Create report"
   - **Report name**: "RAG System Costs"
   - **Time period**: Last 30 days
   - **Granularity**: Daily
   - **Dimensions**: Service
   - **Filters**: Add filter
     - **Tag**: Add tags for RAG resources
   - Click "Save report"

3. **Set Up Cost Budgets**
   - Go to "AWS Budgets"
   - Click "Create budget"
   - **Budget type**: Cost budget
   - **Budget name**: "RAG-Monthly-Budget"
   - **Period**: Monthly
   - **Budgeted amount**: $500 (adjust based on your needs)
   - Click "Next"

4. **Configure Alerts**
   - **Alert threshold**: 80% of budgeted amount
   - **Email recipients**: Enter your email
   - Add another alert at 100%
   - Click "Next"
   - Review and click "Create budget"

5. **Tag Resources for Cost Tracking**
   - Go to each service (SageMaker, OpenSearch, etc.)
   - Add tags:
     - Key: Project, Value: RAG-Production
     - Key: Environment, Value: Production
     - Key: CostCenter, Value: AI-ML


#### Component 14.2: Optimize Resource Usage

**Purpose**: Right-size resources to reduce costs without impacting performance.

**AWS GUI Step-by-Step Instructions**:

1. **Review SageMaker Endpoint Usage**
   - Go to CloudWatch console
   - View metrics for "rag-embedding-endpoint"
   - Check "InvocationsPerInstance" metric
   - If consistently low, consider:
     - Reducing instance count
     - Switching to smaller instance type
     - Using Serverless Inference (if applicable)

2. **Consider SageMaker Serverless Inference**
   - Go to SageMaker console
   - Click "Inference" → "Serverless endpoints"
   - Click "Create endpoint"
   - **Endpoint name**: "rag-embedding-serverless"
   - **Model**: Select your embedding model
   - **Memory size**: 4096 MB
   - **Max concurrency**: 10
   - This charges only for actual usage

3. **Optimize OpenSearch Instance Types**
   - Go to OpenSearch Service console
   - Click on "rag-vector-db"
   - Review "Cluster health" metrics
   - If CPU/Memory consistently low:
     - Click "Actions" → "Edit cluster configuration"
     - Change to smaller instance type
     - Click "Save changes"

4. **Use Bedrock On-Demand Pricing**
   - Already configured (no reserved capacity needed)
   - Monitor usage in CloudWatch
   - Consider switching models if cost is high:
     - Claude 3 Haiku (cheaper, faster)
     - Titan Text Express (most cost-effective)

5. **Implement Request Throttling**
   - In API Gateway, adjust usage plan:
     - Reduce rate limits if not needed
     - Set appropriate quotas
   - This prevents unexpected cost spikes

6. **Schedule Non-Production Resources**
   - For development/testing environments:
   - Create Lambda function to stop/start resources
   - Use EventBridge to schedule:
     - Stop at 6 PM weekdays
     - Start at 8 AM weekdays
     - Off on weekends

## Data Models

### Request/Response Models

#### RAG Query Request
```json
{
  "query": "string (required)",
  "top_k": "integer (optional, default: 5)",
  "temperature": "float (optional, default: 0.7)",
  "max_tokens": "integer (optional, default: 2048)"
}
```

#### RAG Query Response
```json
{
  "answer": "string",
  "sources": ["string"],
  "num_sources": "integer",
  "confidence": "float (optional)",
  "processing_time_ms": "integer"
}
```

#### Document Ingestion Request
```json
{
  "document_id": "string (required)",
  "content": "string (required)",
  "metadata": {
    "source": "string",
    "author": "string",
    "date": "string (ISO 8601)",
    "tags": ["string"]
  }
}
```

### OpenSearch Document Schema

```json
{
  "document_id": "string (keyword)",
  "content": "string (text)",
  "embedding": "array of floats (knn_vector, dimension: 1024)",
  "metadata": {
    "source": "string (keyword)",
    "page": "integer",
    "chunk_index": "integer",
    "timestamp": "date",
    "author": "string (keyword)",
    "tags": "array of strings (keyword)"
  }
}
```

## Error Handling

### Error Response Format

All API errors follow this format:
```json
{
  "error": "string (error message)",
  "error_code": "string (error code)",
  "details": "object (optional additional details)"
}
```

### Common Error Codes

| Error Code | HTTP Status | Description | Resolution |
|------------|-------------|-------------|------------|
| INVALID_QUERY | 400 | Query parameter missing or invalid | Provide valid query string |
| EMBEDDING_FAILED | 500 | Failed to generate embeddings | Check SageMaker endpoint status |
| SEARCH_FAILED | 500 | OpenSearch query failed | Check OpenSearch cluster health |
| GENERATION_FAILED | 500 | Bedrock model invocation failed | Check Bedrock model access |
| GUARDRAIL_BLOCKED | 403 | Content blocked by guardrails | Modify query to comply with policies |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests | Implement exponential backoff |
| AUTHENTICATION_FAILED | 401 | Invalid or missing API key | Provide valid API key in header |

### Error Handling Strategy

1. **Retry Logic**: Implement exponential backoff for transient errors
2. **Circuit Breaker**: Stop requests to failing services temporarily
3. **Fallback Responses**: Provide graceful degradation when services unavailable
4. **Logging**: Log all errors to CloudWatch for debugging
5. **Monitoring**: Set up alarms for error rate thresholds

## Testing Strategy

### Unit Testing

**Scope**: Individual functions and components

**Test Cases**:
1. Embedding generation with various input lengths
2. Vector search with different similarity thresholds
3. Prompt construction with varying context sizes
4. Response parsing and validation
5. Error handling for each component

**Tools**: pytest, moto (for AWS service mocking)

### Integration Testing

**Scope**: End-to-end RAG pipeline

**Test Cases**:
1. Complete query flow from API to response
2. Document ingestion and retrieval
3. Guardrails activation and blocking
4. Cache hit/miss scenarios
5. Auto-scaling behavior under load

**Tools**: Postman, pytest with real AWS services

### Performance Testing

**Scope**: System performance under various loads

**Test Scenarios**:
1. **Baseline**: 10 requests/second for 10 minutes
2. **Peak Load**: 100 requests/second for 5 minutes
3. **Stress Test**: Gradually increase to failure point
4. **Endurance**: Sustained load for 24 hours

**Metrics to Monitor**:
- Response latency (p50, p95, p99)
- Error rate
- Throughput
- Resource utilization (CPU, memory)
- Cost per 1000 requests

**Tools**: Apache JMeter, Locust, AWS CloudWatch

### Security Testing

**Scope**: Vulnerability assessment and penetration testing

**Test Cases**:
1. SQL injection attempts
2. Prompt injection attacks
3. API authentication bypass attempts
4. Rate limiting effectiveness
5. Data encryption verification
6. IAM permission validation

**Tools**: OWASP ZAP, Burp Suite, AWS IAM Access Analyzer

### Acceptance Testing

**Scope**: Validate against business requirements

**Test Cases**:
1. Answer accuracy for domain-specific questions
2. Source attribution correctness
3. Response time within SLA (< 5 seconds)
4. Guardrails blocking inappropriate content
5. Cost per query within budget

**Success Criteria**:
- 90% answer accuracy on test dataset
- 95% of queries complete within 5 seconds
- Zero security vulnerabilities
- Cost < $0.10 per query


## Production Deployment Checklist

### Pre-Deployment

- [ ] All AWS resources created and configured
- [ ] IAM roles and policies reviewed and validated
- [ ] Security groups configured with least-privilege access
- [ ] Secrets stored in Secrets Manager (no hardcoded credentials)
- [ ] VPC and network configuration tested
- [ ] All endpoints and APIs tested in staging environment
- [ ] Monitoring dashboards created and validated
- [ ] CloudWatch alarms configured and tested
- [ ] Backup and disaster recovery procedures documented
- [ ] Cost budgets and alerts configured
- [ ] Security scanning completed (no critical vulnerabilities)
- [ ] Performance testing completed and meets SLAs
- [ ] Documentation completed and reviewed

### Deployment

- [ ] Deploy infrastructure in production VPC
- [ ] Deploy SageMaker endpoints with auto-scaling
- [ ] Deploy OpenSearch domain with snapshots enabled
- [ ] Enable Bedrock models in production account
- [ ] Deploy Lambda functions with proper IAM roles
- [ ] Deploy API Gateway with authentication
- [ ] Enable WAF protection
- [ ] Configure CloudWatch logging and monitoring
- [ ] Test end-to-end RAG pipeline
- [ ] Verify guardrails are active
- [ ] Validate backup procedures
- [ ] Perform smoke tests

### Post-Deployment

- [ ] Monitor system for 24 hours
- [ ] Review CloudWatch metrics and logs
- [ ] Verify auto-scaling behavior
- [ ] Check cost metrics against budget
- [ ] Validate security controls
- [ ] Update documentation with production details
- [ ] Train operations team on monitoring and troubleshooting
- [ ] Schedule regular security reviews
- [ ] Plan for capacity scaling
- [ ] Document lessons learned

## Maintenance and Operations

### Daily Operations

1. **Monitor CloudWatch Dashboard**
   - Check for any alarms
   - Review error rates
   - Monitor latency metrics

2. **Review Logs**
   - Check CloudWatch Logs for errors
   - Review API Gateway access logs
   - Monitor Lambda execution logs

3. **Cost Monitoring**
   - Review daily costs in Cost Explorer
   - Check for any anomalies
   - Verify within budget

### Weekly Operations

1. **Performance Review**
   - Analyze response time trends
   - Review throughput metrics
   - Check resource utilization

2. **Security Review**
   - Review CloudTrail logs
   - Check WAF blocked requests
   - Verify guardrails effectiveness

3. **Capacity Planning**
   - Review auto-scaling events
   - Plan for anticipated traffic changes
   - Adjust thresholds if needed

### Monthly Operations

1. **Cost Optimization**
   - Review detailed cost breakdown
   - Identify optimization opportunities
   - Right-size resources

2. **Backup Verification**
   - Test snapshot restoration
   - Verify backup completeness
   - Update disaster recovery documentation

3. **Security Patching**
   - Update Lambda runtimes
   - Review and update IAM policies
   - Check for AWS service updates

4. **Performance Tuning**
   - Analyze query patterns
   - Optimize OpenSearch indices
   - Fine-tune model parameters

### Quarterly Operations

1. **Comprehensive Security Audit**
   - Penetration testing
   - IAM access review
   - Compliance verification

2. **Disaster Recovery Drill**
   - Full system recovery test
   - Update RTO/RPO metrics
   - Refine recovery procedures

3. **Architecture Review**
   - Evaluate new AWS services
   - Consider architectural improvements
   - Plan major upgrades

## Troubleshooting Guide

### High Latency Issues

**Symptoms**: Response times > 5 seconds

**Diagnosis Steps**:
1. Check CloudWatch metrics for each component
2. Identify bottleneck (embedding, search, or generation)
3. Review CloudWatch Logs for errors

**Solutions**:
- **Embedding Endpoint**: Scale up instances or use larger instance type
- **OpenSearch**: Increase cluster size or optimize queries
- **Bedrock**: Switch to faster model or increase timeout
- **Network**: Check VPC configuration and security groups

### High Error Rates

**Symptoms**: Error rate > 1%

**Diagnosis Steps**:
1. Check CloudWatch Logs for error messages
2. Review API Gateway logs
3. Check service quotas and limits

**Solutions**:
- **Throttling**: Increase service quotas or implement backoff
- **Timeout**: Increase Lambda timeout or optimize code
- **Authentication**: Verify API keys and IAM roles
- **Service Issues**: Check AWS Service Health Dashboard

### Cost Overruns

**Symptoms**: Costs exceeding budget

**Diagnosis Steps**:
1. Review Cost Explorer by service
2. Check CloudWatch metrics for usage patterns
3. Identify unexpected resource usage

**Solutions**:
- **SageMaker**: Reduce instance count or use serverless
- **OpenSearch**: Right-size cluster or reduce replica count
- **Bedrock**: Switch to cheaper model or implement caching
- **Data Transfer**: Optimize data flow and reduce cross-region transfers

### Poor Answer Quality

**Symptoms**: Inaccurate or irrelevant responses

**Diagnosis Steps**:
1. Review retrieved documents for relevance
2. Check embedding quality
3. Analyze prompt construction

**Solutions**:
- **Retrieval**: Adjust top_k parameter or similarity threshold
- **Embeddings**: Fine-tune embedding model on domain data
- **Prompt**: Refine prompt template for better instructions
- **Model**: Switch to more capable foundation model

## Appendix

### Useful AWS CLI Commands

```bash
# Check SageMaker endpoint status
aws sagemaker describe-endpoint --endpoint-name rag-embedding-endpoint

# Invoke SageMaker endpoint
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name rag-embedding-endpoint \
  --body '{"text_inputs":["test"]}' \
  --content-type application/json \
  output.json

# Check OpenSearch domain status
aws opensearch describe-domain --domain-name rag-vector-db

# Invoke Bedrock model
aws bedrock-runtime invoke-model \
  --model-id anthropic.claude-3-sonnet-20240229-v1:0 \
  --body '{"anthropic_version":"bedrock-2023-05-31","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}]}' \
  output.json

# List CloudWatch alarms
aws cloudwatch describe-alarms --alarm-name-prefix RAG

# Get CloudWatch metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name ModelLatency \
  --dimensions Name=EndpointName,Value=rag-embedding-endpoint \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600 \
  --statistics Average
```

### Resource Naming Conventions

| Resource Type | Naming Pattern | Example |
|---------------|----------------|---------|
| VPC | `{project}-{env}-vpc` | rag-production-vpc |
| Subnet | `{project}-{type}-{az}-subnet` | rag-private-1a-subnet |
| Security Group | `{project}-{component}-sg` | rag-sagemaker-sg |
| S3 Bucket | `{project}-{purpose}-{account}-{region}` | rag-documents-123456789012-us-east-1 |
| IAM Role | `{Project}{Component}Role` | RAGSageMakerExecutionRole |
| SageMaker Endpoint | `{project}-{model}-endpoint` | rag-embedding-endpoint |
| OpenSearch Domain | `{project}-{purpose}` | rag-vector-db |
| Lambda Function | `{project}-{purpose}-function` | rag-query-function |
| API Gateway | `{Project}-{Purpose}-API` | RAG-Production-API |
| CloudWatch Alarm | `{Project}-{Component}-{Metric}` | RAG-Embedding-High-Latency |

### Additional Resources

- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [Amazon SageMaker Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/best-practices.html)
- [Amazon OpenSearch Service Best Practices](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/bp.html)
- [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/)
- [RAG Workshop GitHub Repository](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker)

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-01  
**Author**: AWS Solutions Architecture Team  
**Status**: Production Ready


## Phase 15: Web UI Implementation

This phase creates a user-friendly web interface for the RAG application.

### Component 15.1: Create S3 Bucket for Static Website Hosting

**Purpose**: Host the web UI as a static website on S3.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to S3 Service**
   - In AWS Console, go to S3
   - Click "Create bucket"

2. **Configure Bucket for Website**
   - **Bucket name**: "rag-web-ui-[account-id]-[region]"
   - **Region**: Same as your other resources
   - **Block Public Access settings**: UNCHECK "Block all public access"
     - Check the acknowledgment box
   - Click "Create bucket"

3. **Enable Static Website Hosting**
   - Click on the newly created bucket
   - Go to "Properties" tab
   - Scroll to "Static website hosting"
   - Click "Edit"
   - **Static website hosting**: Enable
   - **Hosting type**: Host a static website
   - **Index document**: index.html
   - **Error document**: error.html
   - Click "Save changes"
   - Note the "Bucket website endpoint" URL

4. **Configure Bucket Policy for Public Access**
   - Go to "Permissions" tab
   - Scroll to "Bucket policy"
   - Click "Edit"
   - Paste this policy (replace BUCKET-NAME):
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Sid": "PublicReadGetObject",
         "Effect": "Allow",
         "Principal": "*",
         "Action": "s3:GetObject",
         "Resource": "arn:aws:s3:::BUCKET-NAME/*"
       }
     ]
   }
   ```
   - Click "Save changes"

### Component 15.2: Build Web UI Files

**Purpose**: Create the HTML, CSS, and JavaScript files for the chat interface.

**AWS GUI Step-by-Step Instructions**:

1. **Create index.html File**
   - On your local computer, create a file named `index.html`
   - Copy this code:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Assistant - AI-Powered Q&A</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 RAG Assistant</h1>
            <p class="subtitle">AI-Powered Question Answering System</p>
        </header>

        <div class="chat-container">
            <div id="chatMessages" class="chat-messages">
                <div class="welcome-message">
                    <h2>Welcome! 👋</h2>
                    <p>Ask me anything about the documents in the knowledge base.</p>
                    <p class="hint">Try asking: "What is Amazon SageMaker?"</p>
                </div>
            </div>

            <div class="input-container">
                <form id="queryForm">
                    <div class="input-wrapper">
                        <textarea 
                            id="queryInput" 
                            placeholder="Type your question here..."
                            rows="2"
                            required
                        ></textarea>
                        <button type="submit" id="submitBtn">
                            <span id="btnText">Send</span>
                            <span id="btnLoader" class="loader" style="display: none;"></span>
                        </button>
                    </div>
                </form>
            </div>
        </div>

        <footer>
            <p>Powered by AWS SageMaker, Bedrock, and OpenSearch</p>
        </footer>
    </div>

    <script src="app.js"></script>
</body>
</html>
```

2. **Create styles.css File**
   - Create a file named `styles.css`
   - Copy this code:

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
}

.container {
    width: 100%;
    max-width: 900px;
    background: white;
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    height: 90vh;
    max-height: 800px;
}

header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    text-align: center;
}

header h1 {
    font-size: 2rem;
    margin-bottom: 10px;
}

.subtitle {
    opacity: 0.9;
    font-size: 1rem;
}

.chat-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 30px;
    background: #f8f9fa;
}

.welcome-message {
    text-align: center;
    padding: 40px 20px;
    color: #666;
}

.welcome-message h2 {
    color: #333;
    margin-bottom: 15px;
}

.welcome-message .hint {
    margin-top: 20px;
    padding: 15px;
    background: #e3f2fd;
    border-radius: 10px;
    color: #1976d2;
    font-style: italic;
}

.message {
    margin-bottom: 20px;
    animation: fadeIn 0.3s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.message.user {
    text-align: right;
}

.message-content {
    display: inline-block;
    max-width: 80%;
    padding: 15px 20px;
    border-radius: 15px;
    word-wrap: break-word;
}

.message.user .message-content {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-bottom-right-radius: 5px;
}

.message.assistant .message-content {
    background: white;
    color: #333;
    border: 1px solid #e0e0e0;
    border-bottom-left-radius: 5px;
    text-align: left;
}

.sources {
    margin-top: 15px;
    padding: 15px;
    background: #f0f0f0;
    border-radius: 10px;
    font-size: 0.9rem;
}

.sources h4 {
    color: #666;
    margin-bottom: 10px;
    font-size: 0.85rem;
    text-transform: uppercase;
}

.sources ul {
    list-style: none;
    padding-left: 0;
}

.sources li {
    padding: 5px 0;
    color: #1976d2;
}

.sources li:before {
    content: "📄 ";
    margin-right: 5px;
}

.input-container {
    padding: 20px 30px;
    background: white;
    border-top: 1px solid #e0e0e0;
}

.input-wrapper {
    display: flex;
    gap: 10px;
    align-items: flex-end;
}

textarea {
    flex: 1;
    padding: 15px;
    border: 2px solid #e0e0e0;
    border-radius: 12px;
    font-size: 1rem;
    font-family: inherit;
    resize: none;
    transition: border-color 0.3s;
}

textarea:focus {
    outline: none;
    border-color: #667eea;
}

button {
    padding: 15px 30px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
    min-width: 100px;
}

button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.loader {
    border: 3px solid #f3f3f3;
    border-top: 3px solid #667eea;
    border-radius: 50%;
    width: 20px;
    height: 20px;
    animation: spin 1s linear infinite;
    display: inline-block;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.error-message {
    background: #ffebee;
    color: #c62828;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    border-left: 4px solid #c62828;
}

footer {
    padding: 15px;
    text-align: center;
    background: #f8f9fa;
    color: #666;
    font-size: 0.9rem;
    border-top: 1px solid #e0e0e0;
}

/* Scrollbar styling */
.chat-messages::-webkit-scrollbar {
    width: 8px;
}

.chat-messages::-webkit-scrollbar-track {
    background: #f1f1f1;
}

.chat-messages::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
}

.chat-messages::-webkit-scrollbar-thumb:hover {
    background: #555;
}

/* Responsive design */
@media (max-width: 768px) {
    .container {
        height: 100vh;
        max-height: none;
        border-radius: 0;
    }

    header h1 {
        font-size: 1.5rem;
    }

    .chat-messages {
        padding: 20px;
    }

    .message-content {
        max-width: 90%;
    }
}
```

3. **Create app.js File**
   - Create a file named `app.js`
   - Copy this code (replace API_ENDPOINT and API_KEY):

```javascript
// Configuration - REPLACE THESE VALUES
const API_ENDPOINT = 'https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod/query';
const API_KEY = 'YOUR-API-KEY-HERE';

// DOM elements
const queryForm = document.getElementById('queryForm');
const queryInput = document.getElementById('queryInput');
const chatMessages = document.getElementById('chatMessages');
const submitBtn = document.getElementById('submitBtn');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');

// Initialize
let conversationHistory = [];

// Event listeners
queryForm.addEventListener('submit', handleSubmit);

queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        queryForm.dispatchEvent(new Event('submit'));
    }
});

// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();
    
    const query = queryInput.value.trim();
    if (!query) return;

    // Clear input
    queryInput.value = '';
    
    // Remove welcome message if present
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    // Add user message
    addMessage(query, 'user');

    // Disable input
    setLoading(true);

    try {
        // Call API
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': API_KEY
            },
            body: JSON.stringify({
                query: query,
                top_k: 5
            })
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        
        // Add assistant message
        addMessage(data.answer, 'assistant', data.sources);

        // Store in history
        conversationHistory.push({
            query: query,
            answer: data.answer,
            sources: data.sources
        });

    } catch (error) {
        console.error('Error:', error);
        addErrorMessage('Sorry, something went wrong. Please try again.');
    } finally {
        setLoading(false);
        queryInput.focus();
    }
}

// Add message to chat
function addMessage(text, sender, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;

    messageDiv.appendChild(contentDiv);

    // Add sources if available
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        sourcesDiv.innerHTML = `
            <h4>📚 Sources:</h4>
            <ul>
                ${sources.map(source => `<li>${source}</li>`).join('')}
            </ul>
        `;
        contentDiv.appendChild(sourcesDiv);
    }

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Add error message
function addErrorMessage(text) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = text;
    chatMessages.appendChild(errorDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Set loading state
function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    queryInput.disabled = isLoading;
    
    if (isLoading) {
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-block';
    } else {
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

// Auto-resize textarea
queryInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 150) + 'px';
});
```

4. **Create config.js File** (for easy configuration)
   - Create a file named `config.js`
   - Copy this code:

```javascript
// RAG Application Configuration
// Update these values after deploying your AWS infrastructure

const CONFIG = {
    // Your API Gateway endpoint URL
    API_ENDPOINT: 'https://YOUR-API-ID.execute-api.REGION.amazonaws.com/prod/query',
    
    // Your API Gateway API Key
    API_KEY: 'YOUR-API-KEY-HERE',
    
    // Optional: Customize these settings
    MAX_SOURCES: 5,
    TIMEOUT_MS: 30000,
    RETRY_ATTEMPTS: 2
};

// Export for use in app.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
}
```

### Component 15.3: Upload UI Files to S3

**Purpose**: Deploy the web UI to S3.

**AWS GUI Step-by-Step Instructions**:

1. **Upload Files to S3**
   - Go to S3 console
   - Click on "rag-web-ui-..." bucket
   - Click "Upload" button
   - Click "Add files"
   - Select all files: index.html, styles.css, app.js, config.js
   - Click "Upload"
   - Wait for upload to complete

2. **Update app.js with Your API Details**
   - Before uploading, edit app.js
   - Replace `API_ENDPOINT` with your API Gateway URL
   - Replace `API_KEY` with your API key
   - Save and re-upload

3. **Test the Website**
   - Go to bucket "Properties" tab
   - Find "Static website hosting" section
   - Click the "Bucket website endpoint" URL
   - Your RAG UI should load!

### Component 15.4: Set Up CloudFront for HTTPS (Optional but Recommended)

**Purpose**: Add HTTPS and improve performance with CDN.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to CloudFront**
   - In AWS Console, search for "CloudFront"
   - Click "Create distribution"

2. **Configure Origin**
   - **Origin domain**: Paste your S3 website endpoint (NOT the bucket name)
   - **Protocol**: HTTP only (S3 website endpoints don't support HTTPS)
   - **Name**: Leave default

3. **Configure Default Cache Behavior**
   - **Viewer protocol policy**: Redirect HTTP to HTTPS
   - **Allowed HTTP methods**: GET, HEAD
   - **Cache policy**: CachingOptimized
   - Leave other settings as default

4. **Configure Settings**
   - **Price class**: Use all edge locations (or select based on your needs)
   - **Alternate domain name (CNAME)**: Leave empty (or add your custom domain)
   - **Default root object**: index.html

5. **Create Distribution**
   - Click "Create distribution"
   - Wait 5-10 minutes for deployment
   - Status will change from "Deploying" to "Enabled"

6. **Access Your UI**
   - Copy the "Distribution domain name" (e.g., d123abc.cloudfront.net)
   - Open in browser: https://d123abc.cloudfront.net
   - Your RAG UI is now available over HTTPS!

7. **Update CORS in API Gateway** (if needed)
   - Go to API Gateway console
   - Select your RAG API
   - Click on /query resource
   - Click OPTIONS method
   - Update CORS to allow CloudFront domain

### UI Features

Your RAG web UI includes:

✅ **Modern Chat Interface**: Clean, responsive design  
✅ **Real-time Responses**: Streaming-like experience  
✅ **Source Citations**: Shows which documents were used  
✅ **Error Handling**: User-friendly error messages  
✅ **Loading States**: Visual feedback during processing  
✅ **Mobile Responsive**: Works on all devices  
✅ **Keyboard Shortcuts**: Enter to send, Shift+Enter for new line  
✅ **Auto-scroll**: Automatically scrolls to latest message  

### Testing the UI

1. Open the website URL
2. Type a question like "What is Amazon SageMaker?"
3. Click "Send" or press Enter
4. Wait for the response (3-5 seconds)
5. Review the answer and source documents
6. Ask follow-up questions

### Troubleshooting UI Issues

**Issue**: "API error: 403"
- **Solution**: Check that API key is correct in app.js

**Issue**: "CORS error"
- **Solution**: Enable CORS in API Gateway for your domain

**Issue**: "Network error"
- **Solution**: Verify API Gateway endpoint URL is correct

**Issue**: Blank page
- **Solution**: Check browser console for JavaScript errors

**Issue**: Slow responses
- **Solution**: Check Lambda timeout and CloudWatch logs
