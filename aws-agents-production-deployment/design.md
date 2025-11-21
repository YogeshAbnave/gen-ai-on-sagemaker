# Design Document: Production-Grade AI Agents on AWS

## Overview

This design document provides comprehensive AWS GUI step-by-step instructions for deploying a production-grade AI Agents application using Amazon SageMaker AI, Amazon Bedrock, AWS Lambda, and Step Functions. The design follows AWS Well-Architected Framework principles and implements autonomous agents capable of reasoning, tool usage, multi-step workflows, and collaborative problem-solving.

The AI Agents system architecture consists of:
- **Agent Layer**: Bedrock models with tool calling, agent frameworks (LangGraph, CrewAI)
- **Orchestration Layer**: Step Functions for workflows, EventBridge for triggers
- **Compute Layer**: Lambda for agent execution, SageMaker for development
- **State Layer**: DynamoDB for memory, S3 for artifacts
- **Tool Layer**: Custom functions, API integrations, database connectors
- **Observability Layer**: X-Ray for tracing, LangFuse for agent monitoring, CloudWatch for metrics
- **Security Layer**: IAM roles, Guardrails, Secrets Manager, VPC isolation

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
│  │                               │  │   Lambda   │ │            │ │
│  │                               │  │  (Agents)  │ │            │ │
│  │                               │  └────────────┘ │            │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  DynamoDB    │  │   Bedrock    │  │ Step Functions│             │
│  │  (Memory)    │  │  (Models)    │  │ (Workflows)  │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ API Gateway  │  │  EventBridge │  │   X-Ray      │              │
│  │  (REST/WS)   │  │  (Triggers)  │  │  (Tracing)   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

### Agent Execution Flow

```
User Request → API Gateway → Lambda (Agent Orchestrator)
                                   ↓
                          [Load Agent State from DynamoDB]
                                   ↓
                          [Bedrock: Reasoning & Planning]
                                   ↓
                          [Tool Call Decision]
                                   ↓
                    ┌──────────────┴──────────────┐
                    ↓                             ↓
            [Execute Tool]                [No Tool Needed]
                    ↓                             ↓
            [Process Result]              [Generate Response]
                    ↓                             ↓
                    └──────────────┬──────────────┘
                                   ↓
                          [Save State to DynamoDB]
                                   ↓
                          [Return Response]
                                   ↓
                          Response → User
```


## Components and Interfaces

### Phase 1: Foundation Infrastructure Setup

This phase establishes the foundational AWS infrastructure for AI agents.

#### Component 1.1: VPC and Network Configuration

**Purpose**: Create an isolated network environment for agent deployment.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to VPC Service**
   - Log into AWS Console (https://console.aws.amazon.com)
   - Search for "VPC" in the top search bar
   - Click on "VPC"

2. **Launch VPC Creation Wizard**
   - Click "Create VPC" button
   - Select "VPC and more"

3. **Configure VPC Settings**
   - **Name tag**: "agents-production-vpc"
   - **IPv4 CIDR**: "10.0.0.0/16"
   - **IPv6**: No IPv6 CIDR block
   - **Tenancy**: Default

4. **Configure Subnets**
   - **Availability Zones**: 2
   - **Public subnets**: 2
   - **Private subnets**: 2
   - **NAT gateways**: 1 per AZ
   - Click "Create VPC"

5. **Note VPC Details**
   - Save VPC ID, subnet IDs, security group ID

#### Component 1.2: DynamoDB Tables for Agent State

**Purpose**: Create tables to store agent conversation history and state.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to DynamoDB**
   - Search for "DynamoDB" in AWS Console
   - Click "DynamoDB"

2. **Create Agent Sessions Table**
   - Click "Create table"
   - **Table name**: "agent-sessions"
   - **Partition key**: "session_id" (String)
   - **Sort key**: "timestamp" (Number)
   - **Table settings**: On-demand capacity
   - **Encryption**: AWS owned key
   - Click "Create table"

3. **Create Agent Memory Table**
   - Click "Create table" again
   - **Table name**: "agent-memory"
   - **Partition key**: "user_id" (String)
   - **Sort key**: "conversation_id" (String)
   - **Table settings**: On-demand capacity
   - Click "Create table"

4. **Create Agent Workflows Table**
   - Click "Create table"
   - **Table name**: "agent-workflows"
   - **Partition key**: "workflow_id" (String)
   - **Sort key**: "step_number" (Number)
   - **Table settings**: On-demand capacity
   - Click "Create table"

5. **Enable Point-in-Time Recovery**
   - For each table, go to "Backups" tab
   - Click "Edit" under Point-in-time recovery
   - Enable PITR
   - Click "Save changes"

6. **Note Table Details**
   - Save table names and ARNs

#### Component 1.3: S3 Buckets for Agent Artifacts

**Purpose**: Store agent configurations, tool definitions, and execution logs.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to S3**
   - Search for "S3"
   - Click "S3"

2. **Create Agent Configurations Bucket**
   - Click "Create bucket"
   - **Name**: "agents-config-[account-id]-[region]"
   - **Region**: Your preferred region
   - **Block Public Access**: Keep all checked
   - **Versioning**: Enable
   - **Encryption**: SSE-S3
   - Click "Create bucket"

3. **Create Agent Logs Bucket**
   - Click "Create bucket"
   - **Name**: "agents-logs-[account-id]-[region]"
   - Follow same settings
   - Click "Create bucket"

4. **Create Agent Tools Bucket**
   - Click "Create bucket"
   - **Name**: "agents-tools-[account-id]-[region]"
   - Follow same settings
   - Click "Create bucket"

5. **Create Folder Structure**
   - In agents-config bucket, create folders:
     - "tool-definitions"
     - "agent-prompts"
     - "workflow-definitions"

#### Component 1.4: IAM Roles for Agents

**Purpose**: Create IAM roles with permissions for agent execution.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to IAM**
   - Search for "IAM"
   - Click "IAM"

2. **Create Agent Execution Role**
   - Click "Roles" → "Create role"
   - **Trusted entity**: AWS service
   - **Use case**: Lambda
   - Click "Next"

3. **Attach Policies**
   - Search and select:
     - ✓ AWSLambdaBasicExecutionRole
     - ✓ AmazonBedrockFullAccess
     - ✓ AmazonDynamoDBFullAccess
     - ✓ AmazonS3ReadOnlyAccess
   - Click "Next"

4. **Name the Role**
   - **Role name**: "AgentExecutionRole"
   - **Description**: "Execution role for AI agents"
   - Click "Create role"

5. **Add Inline Policy for X-Ray**
   - Click on "AgentExecutionRole"
   - Click "Add permissions" → "Create inline policy"
   - Click "JSON" tab
   - Paste:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "xray:PutTraceSegments",
           "xray:PutTelemetryRecords"
         ],
         "Resource": "*"
       }
     ]
   }
   ```
   - **Policy name**: "XRayTracingPolicy"
   - Click "Create policy"

6. **Create Step Functions Role**
   - Click "Create role"
   - **Trusted entity**: AWS service
   - **Use case**: Step Functions
   - Attach policies:
     - ✓ AWSLambdaRole
     - ✓ CloudWatchLogsFullAccess
   - **Role name**: "AgentWorkflowRole"
   - Click "Create role"

7. **Note Role ARNs**
   - Save AgentExecutionRole ARN
   - Save AgentWorkflowRole ARN


### Phase 2: SageMaker Studio Setup for Agent Development

This phase sets up the development environment with agent frameworks.

#### Component 2.1: Create SageMaker Domain

**Purpose**: Set up SageMaker Studio for agent development and testing.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to SageMaker**
   - Search for "SageMaker"
   - Click "Amazon SageMaker"

2. **Create Domain**
   - Click "Domains" in left sidebar
   - Click "Create domain"
   - Select "Set up for organization"
   - Click "Set up"

3. **Configure Domain**
   - **Domain name**: "agents-development-domain"
   - **Authentication**: IAM
   - **VPC**: Select "agents-production-vpc"
   - **Subnets**: Select private subnets
   - **Security groups**: Create new or select existing
   - **Execution role**: Create new role or select "AgentExecutionRole"
   - Click "Submit"

4. **Wait for Domain Creation**
   - Takes 5-10 minutes
   - Status will show "InService"

5. **Create User Profile**
   - Click on domain name
   - Click "Add user"
   - **Name**: "agent-developer"
   - **Execution role**: AgentExecutionRole
   - Click "Submit"

6. **Launch Studio**
   - Click on user profile
   - Click "Launch" → "Studio"
   - Wait for Studio to load

#### Component 2.2: Install Agent Frameworks

**Purpose**: Install LangGraph, CrewAI, and other agent frameworks.

**AWS GUI Step-by-Step Instructions**:

1. **In SageMaker Studio**
   - Click "File" → "New" → "Terminal"

2. **Install Agent Frameworks**
   - Run these commands in terminal:
   ```bash
   pip install langgraph langchain langchain-aws langchain-community
   pip install crewai crewai-tools
   pip install strands-agents
   pip install boto3 anthropic openai
   pip install langfuse
   ```

3. **Clone Agent Workshop**
   - In terminal:
   ```bash
   git clone https://github.com/aws-samples/generative-ai-on-amazon-sagemaker.git
   cd generative-ai-on-amazon-sagemaker/workshops/diy-agents-with-sagemaker-and-bedrock
   ```

4. **Install Workshop Dependencies**
   - Run:
   ```bash
   pip install -r 0-setup/requirements.txt
   ```

5. **Verify Installation**
   - Create new notebook
   - Run:
   ```python
   import langgraph
   import crewai
   import boto3
   print("All frameworks installed successfully!")
   ```

### Phase 3: Enable Bedrock Models with Tool Calling

This phase enables foundation models that support tool calling.

#### Component 3.1: Enable Bedrock Models

**Purpose**: Request access to models with tool calling capabilities.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to Bedrock**
   - Search for "Bedrock"
   - Click "Amazon Bedrock"

2. **Access Model Access Page**
   - Click "Model access" in left sidebar

3. **Request Model Access**
   - Click "Manage model access"
   - Select these models:
     - ✓ Claude 3.5 Sonnet (best for agents)
     - ✓ Claude 3 Sonnet
     - ✓ Claude 3 Haiku (cost-effective)
   - Click "Request model access"

4. **Wait for Approval**
   - Most models are instantly available
   - Status will show "Access granted"

5. **Test Tool Calling in Playground**
   - Click "Playgrounds" → "Chat"
   - Select "Claude 3.5 Sonnet"
   - Click "Advanced settings"
   - Scroll to "Tools"
   - Click "Add tool"

6. **Define Test Tool**
   - **Tool name**: "get_weather"
   - **Description**: "Get current weather for a location"
   - **Parameters**:
     ```json
     {
       "type": "object",
       "properties": {
         "location": {
           "type": "string",
           "description": "City name"
         }
       },
       "required": ["location"]
     }
     ```
   - Click "Add tool"

7. **Test Tool Calling**
   - In chat, type: "What's the weather in Seattle?"
   - Model should return a tool call request
   - Verify tool call format

8. **Note Model IDs**
   - Claude 3.5 Sonnet: anthropic.claude-3-5-sonnet-20240620-v1:0
   - Claude 3 Sonnet: anthropic.claude-3-sonnet-20240229-v1:0

### Phase 4: Implement Basic Agent Patterns

This phase implements fundamental agent patterns like ReAct and tool calling.

#### Component 4.1: Create ReAct Agent Notebook

**Purpose**: Build a reasoning and acting agent.

**AWS GUI Step-by-Step Instructions**:

1. **Create New Notebook in Studio**
   - File → New → Notebook
   - Kernel: Python 3 (Data Science 3.0)

2. **Setup and Configuration**
   - First cell:
   ```python
   import boto3
   import json
   from typing import List, Dict, Any
   
   # Initialize Bedrock client
   bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
   
   # Configuration
   MODEL_ID = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
   ```

3. **Define Tool Functions**
   - New cell:
   ```python
   def get_current_time():
       """Get the current time"""
       from datetime import datetime
       return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   
   def calculate(expression: str):
       """Calculate a mathematical expression"""
       try:
           result = eval(expression)
           return f"Result: {result}"
       except Exception as e:
           return f"Error: {str(e)}"
   
   def search_database(query: str):
       """Search a mock database"""
       # Mock implementation
       database = {
           "AWS": "Amazon Web Services is a cloud platform",
           "SageMaker": "Amazon SageMaker is a machine learning service",
           "Bedrock": "Amazon Bedrock provides foundation models"
       }
       return database.get(query, "No information found")
   
   # Tool registry
   TOOLS = {
       "get_current_time": get_current_time,
       "calculate": calculate,
       "search_database": search_database
   }
   ```

4. **Define Tool Schemas**
   - New cell:
   ```python
   TOOL_SCHEMAS = [
       {
           "name": "get_current_time",
           "description": "Get the current date and time",
           "input_schema": {
               "type": "object",
               "properties": {},
               "required": []
           }
       },
       {
           "name": "calculate",
           "description": "Calculate a mathematical expression",
           "input_schema": {
               "type": "object",
               "properties": {
                   "expression": {
                       "type": "string",
                       "description": "Mathematical expression to evaluate"
                   }
               },
               "required": ["expression"]
           }
       },
       {
           "name": "search_database",
           "description": "Search the knowledge database",
           "input_schema": {
               "type": "object",
               "properties": {
                   "query": {
                       "type": "string",
                       "description": "Search query"
                   }
               },
               "required": ["query"]
           }
       }
   ]
   ```

5. **Implement ReAct Agent**
   - New cell:
   ```python
   def react_agent(user_query: str, max_iterations: int = 5):
       """
       ReAct agent: Reasoning and Acting
       """
       conversation_history = []
       
       # Initial user message
       conversation_history.append({
           "role": "user",
           "content": user_query
       })
       
       for iteration in range(max_iterations):
           print(f"\n--- Iteration {iteration + 1} ---")
           
           # Call Bedrock with tools
           body = json.dumps({
               "anthropic_version": "bedrock-2023-05-31",
               "max_tokens": 2048,
               "messages": conversation_history,
               "tools": TOOL_SCHEMAS
           })
           
           response = bedrock_runtime.invoke_model(
               modelId=MODEL_ID,
               body=body
           )
           
           response_body = json.loads(response['body'].read())
           
           # Check stop reason
           stop_reason = response_body.get('stop_reason')
           
           if stop_reason == 'end_turn':
               # Agent finished
               final_response = response_body['content'][0]['text']
               print(f"Final Answer: {final_response}")
               return final_response
           
           elif stop_reason == 'tool_use':
               # Agent wants to use a tool
               content_blocks = response_body['content']
               
               # Add assistant response to history
               conversation_history.append({
                   "role": "assistant",
                   "content": content_blocks
               })
               
               # Execute tools
               tool_results = []
               for block in content_blocks:
                   if block['type'] == 'tool_use':
                       tool_name = block['name']
                       tool_input = block['input']
                       tool_id = block['id']
                       
                       print(f"Tool Call: {tool_name}({tool_input})")
                       
                       # Execute tool
                       if tool_name in TOOLS:
                           result = TOOLS[tool_name](**tool_input)
                           print(f"Tool Result: {result}")
                           
                           tool_results.append({
                               "type": "tool_result",
                               "tool_use_id": tool_id,
                               "content": str(result)
                           })
               
               # Add tool results to history
               conversation_history.append({
                   "role": "user",
                   "content": tool_results
               })
           
           else:
               print(f"Unexpected stop reason: {stop_reason}")
               break
       
       return "Max iterations reached"
   ```

6. **Test ReAct Agent**
   - New cell:
   ```python
   # Test 1: Simple calculation
   result = react_agent("What is 25 * 47?")
   print("\n" + "="*80 + "\n")
   
   # Test 2: Time query
   result = react_agent("What time is it now?")
   print("\n" + "="*80 + "\n")
   
   # Test 3: Database search
   result = react_agent("Tell me about Amazon SageMaker")
   print("\n" + "="*80 + "\n")
   
   # Test 4: Multi-step reasoning
   result = react_agent("What is 100 divided by 4, and then multiply that by 3?")
   ```

7. **Save Notebook**
   - File → Save Notebook As...
   - Name: "react-agent-implementation.ipynb"


### Phase 5: Implement Agent Memory with DynamoDB

This phase adds persistent memory to agents for maintaining conversation context.

#### Component 5.1: Create Memory Management Functions

**Purpose**: Store and retrieve agent conversation history.

**AWS GUI Step-by-Step Instructions**:

1. **Create Memory Notebook in Studio**
   - File → New → Notebook

2. **Setup DynamoDB Client**
   ```python
   import boto3
   from datetime import datetime
   import uuid
   
   dynamodb = boto3.resource('dynamodb')
   memory_table = dynamodb.Table('agent-memory')
   sessions_table = dynamodb.Table('agent-sessions')
   ```

3. **Implement Memory Functions**
   ```python
   def save_message(user_id: str, conversation_id: str, role: str, content: str):
       """Save a message to agent memory"""
       timestamp = int(datetime.now().timestamp() * 1000)
       
       memory_table.put_item(Item={
           'user_id': user_id,
           'conversation_id': conversation_id,
           'timestamp': timestamp,
           'role': role,
           'content': content,
           'message_id': str(uuid.uuid4())
       })
   
   def get_conversation_history(user_id: str, conversation_id: str, limit: int = 10):
       """Retrieve conversation history"""
       response = memory_table.query(
           KeyConditionExpression='user_id = :uid AND conversation_id = :cid',
           ExpressionAttributeValues={
               ':uid': user_id,
               ':cid': conversation_id
           },
           Limit=limit,
           ScanIndexForward=False  # Most recent first
       )
       
       messages = response['Items']
       messages.reverse()  # Chronological order
       return messages
   
   def create_session(user_id: str):
       """Create a new agent session"""
       session_id = str(uuid.uuid4())
       timestamp = int(datetime.now().timestamp() * 1000)
       
       sessions_table.put_item(Item={
           'session_id': session_id,
           'timestamp': timestamp,
           'user_id': user_id,
           'status': 'active'
       })
       
       return session_id
   ```

4. **Integrate Memory with Agent**
   ```python
   def agent_with_memory(user_id: str, conversation_id: str, user_query: str):
       """Agent with persistent memory"""
       
       # Load conversation history
       history = get_conversation_history(user_id, conversation_id)
       
       # Convert to Bedrock format
       messages = []
       for msg in history:
           messages.append({
               "role": msg['role'],
               "content": msg['content']
           })
       
       # Add new user message
       messages.append({
           "role": "user",
           "content": user_query
       })
       
       # Save user message
       save_message(user_id, conversation_id, "user", user_query)
       
       # Call Bedrock
       body = json.dumps({
           "anthropic_version": "bedrock-2023-05-31",
           "max_tokens": 2048,
           "messages": messages
       })
       
       response = bedrock_runtime.invoke_model(
           modelId=MODEL_ID,
           body=body
       )
       
       response_body = json.loads(response['body'].read())
       assistant_message = response_body['content'][0]['text']
       
       # Save assistant message
       save_message(user_id, conversation_id, "assistant", assistant_message)
       
       return assistant_message
   ```

5. **Test Memory System**
   ```python
   user_id = "user123"
   conversation_id = "conv456"
   
   # First message
   response1 = agent_with_memory(user_id, conversation_id, "My name is Alice")
   print(f"Agent: {response1}")
   
   # Second message (agent should remember name)
   response2 = agent_with_memory(user_id, conversation_id, "What's my name?")
   print(f"Agent: {response2}")
   ```

### Phase 6: Deploy Agent as Lambda Function

This phase packages the agent for serverless deployment.

#### Component 6.1: Create Lambda Function for Agent

**Purpose**: Deploy agent as a scalable Lambda function.

**AWS GUI Step-by-Step Instructions**:

1. **Navigate to Lambda**
   - Search for "Lambda"
   - Click "AWS Lambda"

2. **Create Function**
   - Click "Create function"
   - Select "Author from scratch"
   - **Function name**: "ai-agent-executor"
   - **Runtime**: Python 3.11
   - **Architecture**: x86_64
   - **Execution role**: Use existing role → AgentExecutionRole
   - Click "Create function"

3. **Configure Function Settings**
   - Click "Configuration" tab
   - Click "General configuration" → "Edit"
   - **Memory**: 2048 MB
   - **Timeout**: 5 minutes
   - **Ephemeral storage**: 1024 MB
   - Click "Save"

4. **Add Environment Variables**
   - Click "Environment variables" → "Edit"
   - Add:
     - MODEL_ID = anthropic.claude-3-5-sonnet-20240620-v1:0
     - MEMORY_TABLE = agent-memory
     - SESSIONS_TABLE = agent-sessions
     - REGION = us-east-1
   - Click "Save"

5. **Enable X-Ray Tracing**
   - Click "Monitoring and operations tools"
   - Click "Edit"
   - **Active tracing**: Enable
   - Click "Save"

6. **Add Lambda Layer for Dependencies**
   - Click "Code" tab
   - Scroll to "Layers"
   - Click "Add a layer"
   - Select "Specify an ARN"
   - Use AWS SDK layer or create custom layer with:
     - boto3, anthropic, langchain
   - Click "Add"

7. **Write Lambda Handler Code**
   - Replace code with:
   ```python
   import json
   import boto3
   import os
   from datetime import datetime
   import uuid
   
   # Initialize clients
   bedrock_runtime = boto3.client('bedrock-runtime')
   dynamodb = boto3.resource('dynamodb')
   
   # Configuration
   MODEL_ID = os.environ['MODEL_ID']
   MEMORY_TABLE = os.environ['MEMORY_TABLE']
   memory_table = dynamodb.Table(MEMORY_TABLE)
   
   # Tool definitions
   TOOL_SCHEMAS = [
       {
           "name": "get_current_time",
           "description": "Get the current date and time",
           "input_schema": {
               "type": "object",
               "properties": {},
               "required": []
           }
       }
   ]
   
   def get_current_time():
       return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   
   TOOLS = {
       "get_current_time": get_current_time
   }
   
   def save_message(user_id, conversation_id, role, content):
       timestamp = int(datetime.now().timestamp() * 1000)
       memory_table.put_item(Item={
           'user_id': user_id,
           'conversation_id': conversation_id,
           'timestamp': timestamp,
           'role': role,
           'content': content,
           'message_id': str(uuid.uuid4())
       })
   
   def get_conversation_history(user_id, conversation_id, limit=10):
       response = memory_table.query(
           KeyConditionExpression='user_id = :uid AND conversation_id = :cid',
           ExpressionAttributeValues={
               ':uid': user_id,
               ':cid': conversation_id
           },
           Limit=limit,
           ScanIndexForward=False
       )
       messages = response['Items']
       messages.reverse()
       return messages
   
   def execute_agent(user_id, conversation_id, user_query, max_iterations=5):
       # Load history
       history = get_conversation_history(user_id, conversation_id)
       messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
       messages.append({"role": "user", "content": user_query})
       
       # Save user message
       save_message(user_id, conversation_id, "user", user_query)
       
       # Agent loop
       for iteration in range(max_iterations):
           body = json.dumps({
               "anthropic_version": "bedrock-2023-05-31",
               "max_tokens": 2048,
               "messages": messages,
               "tools": TOOL_SCHEMAS
           })
           
           response = bedrock_runtime.invoke_model(
               modelId=MODEL_ID,
               body=body
           )
           
           response_body = json.loads(response['body'].read())
           stop_reason = response_body.get('stop_reason')
           
           if stop_reason == 'end_turn':
               final_response = response_body['content'][0]['text']
               save_message(user_id, conversation_id, "assistant", final_response)
               return final_response
           
           elif stop_reason == 'tool_use':
               content_blocks = response_body['content']
               messages.append({"role": "assistant", "content": content_blocks})
               
               tool_results = []
               for block in content_blocks:
                   if block['type'] == 'tool_use':
                       tool_name = block['name']
                       tool_input = block['input']
                       tool_id = block['id']
                       
                       if tool_name in TOOLS:
                           result = TOOLS[tool_name](**tool_input)
                           tool_results.append({
                               "type": "tool_result",
                               "tool_use_id": tool_id,
                               "content": str(result)
                           })
               
               messages.append({"role": "user", "content": tool_results})
       
       return "Max iterations reached"
   
   def lambda_handler(event, context):
       try:
           # Parse request
           body = json.loads(event['body']) if isinstance(event.get('body'), str) else event
           
           user_id = body.get('user_id', 'anonymous')
           conversation_id = body.get('conversation_id', str(uuid.uuid4()))
           query = body.get('query', '')
           
           if not query:
               return {
                   'statusCode': 400,
                   'body': json.dumps({'error': 'Query is required'})
               }
           
           # Execute agent
           response = execute_agent(user_id, conversation_id, query)
           
           return {
               'statusCode': 200,
               'headers': {
                   'Content-Type': 'application/json',
                   'Access-Control-Allow-Origin': '*'
               },
               'body': json.dumps({
                   'response': response,
                   'conversation_id': conversation_id,
                   'user_id': user_id
               })
           }
           
       except Exception as e:
           print(f"Error: {str(e)}")
           return {
               'statusCode': 500,
               'body': json.dumps({'error': str(e)})
           }
   ```
   - Click "Deploy"

8. **Test Lambda Function**
   - Click "Test" tab
   - Create test event:
   ```json
   {
     "body": "{\"user_id\": \"test-user\", \"conversation_id\": \"test-conv\", \"query\": \"What time is it?\"}"
   }
   ```
   - Click "Test"
   - Review execution results

9. **Note Lambda Details**
   - Function ARN
   - Function name


### Phase 7: Create API Gateway for Agent Access

**Purpose**: Expose agent Lambda through REST API.

**AWS GUI Steps** (Abbreviated):

1. Navigate to API Gateway → Create REST API
2. **API name**: "AI-Agents-API"
3. Create resource: `/agent`
4. Create POST method → Lambda integration → Select "ai-agent-executor"
5. Enable CORS
6. Create API key and usage plan
7. Deploy to "prod" stage
8. Test endpoint:
```bash
curl -X POST https://[api-id].execute-api.us-east-1.amazonaws.com/prod/agent \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_KEY" \
  -d '{"user_id": "user123", "query": "Hello!"}'
```

### Phase 8: Implement Step Functions for Multi-Agent Workflows

**Purpose**: Orchestrate complex multi-step agent workflows.

**AWS GUI Steps**:

1. Navigate to Step Functions → Create state machine
2. **Name**: "multi-agent-workflow"
3. **Type**: Standard
4. **Definition**:
```json
{
  "Comment": "Multi-agent workflow orchestration",
  "StartAt": "ResearchAgent",
  "States": {
    "ResearchAgent": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:ai-agent-executor",
      "Parameters": {
        "body": {
          "user_id.$": "$.user_id",
          "query.$": "$.research_query",
          "agent_type": "researcher"
        }
      },
      "ResultPath": "$.research_result",
      "Next": "AnalysisAgent"
    },
    "AnalysisAgent": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:ai-agent-executor",
      "Parameters": {
        "body": {
          "user_id.$": "$.user_id",
          "query.$": "$.research_result.response",
          "agent_type": "analyst"
        }
      },
      "ResultPath": "$.analysis_result",
      "Next": "SummaryAgent"
    },
    "SummaryAgent": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:ai-agent-executor",
      "Parameters": {
        "body": {
          "user_id.$": "$.user_id",
          "query.$": "$.analysis_result.response",
          "agent_type": "summarizer"
        }
      },
      "ResultPath": "$.final_result",
      "End": true
    }
  }
}
```
5. **Execution role**: AgentWorkflowRole
6. Click "Create"
7. Test execution with input:
```json
{
  "user_id": "user123",
  "research_query": "Research AWS AI services"
}
```

### Phase 9: Implement Observability with X-Ray and LangFuse

**Purpose**: Monitor and trace agent execution.

**AWS GUI Steps**:

1. **Enable X-Ray** (already done in Lambda)
2. Navigate to X-Ray → Service map
3. View agent execution traces
4. **Set up LangFuse**:
   - Sign up at langfuse.com
   - Get API keys
   - Add to Lambda environment variables:
     - LANGFUSE_PUBLIC_KEY
     - LANGFUSE_SECRET_KEY
     - LANGFUSE_HOST
5. Update Lambda code to include LangFuse tracing:
```python
from langfuse import Langfuse

langfuse = Langfuse()

@langfuse.observe()
def execute_agent(...):
    # Agent code
    pass
```

### Phase 10: Implement Bedrock Guardrails for Agents

**Purpose**: Add safety controls to agent behavior.

**AWS GUI Steps**:

1. Navigate to Bedrock → Guardrails
2. Create guardrail: "agent-safety-guardrail"
3. Configure:
   - Content filters: HIGH for all categories
   - Denied topics: Add sensitive topics
   - Word filters: Add inappropriate terms
   - **Contextual grounding**: Enable (prevents hallucinations)
4. Create version
5. Update Lambda code to use guardrail:
```python
response = bedrock_runtime.invoke_model(
    modelId=MODEL_ID,
    body=body,
    guardrailIdentifier='guardrail-id',
    guardrailVersion='1'
)
```

### Phase 11: Implement Real-World Use Case - Text-to-SQL Agent

**Purpose**: Build a practical agent that converts natural language to SQL.

**AWS GUI Steps**:

1. **Create RDS Database** (or use Athena)
   - Navigate to RDS → Create database
   - Engine: PostgreSQL
   - Template: Free tier
   - DB name: "sample-ecommerce"
   - Create sample tables

2. **Create SQL Tool in Lambda**:
```python
import psycopg2

def execute_sql(query: str):
    """Execute SQL query safely"""
    conn = psycopg2.connect(
        host=os.environ['DB_HOST'],
        database=os.environ['DB_NAME'],
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD']
    )
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results

TOOL_SCHEMAS.append({
    "name": "execute_sql",
    "description": "Execute a SQL query on the database",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query to execute"
            }
        },
        "required": ["query"]
    }
})

TOOLS["execute_sql"] = execute_sql
```

3. **Test Text-to-SQL Agent**:
```json
{
  "query": "Show me the top 5 customers by total purchase amount"
}
```

### Phase 12: Build Web UI for Agent Interaction

**Purpose**: Create user-friendly interface for agents.

**AWS GUI Steps**:

1. Create S3 bucket: "agents-web-ui-[account-id]-[region]"
2. Enable static website hosting
3. Create `index.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <title>AI Agent Assistant</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; }
        #chat { border: 1px solid #ccc; height: 400px; overflow-y: scroll; padding: 20px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background: #e3f2fd; text-align: right; }
        .agent { background: #f5f5f5; }
        .thinking { color: #666; font-style: italic; }
        input { width: 80%; padding: 10px; }
        button { padding: 10px 20px; }
    </style>
</head>
<body>
    <h1>🤖 AI Agent Assistant</h1>
    <div id="chat"></div>
    <div id="thinking" class="thinking" style="display:none;">Agent is thinking...</div>
    <input type="text" id="query" placeholder="Ask me anything...">
    <button onclick="sendQuery()">Send</button>
    
    <script>
        const API_ENDPOINT = 'https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod/agent';
        const API_KEY = 'YOUR-API-KEY';
        let conversationId = generateUUID();
        
        function generateUUID() {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        }
        
        async function sendQuery() {
            const query = document.getElementById('query').value;
            if (!query) return;
            
            // Display user message
            addMessage(query, 'user');
            document.getElementById('query').value = '';
            document.getElementById('thinking').style.display = 'block';
            
            try {
                const response = await fetch(API_ENDPOINT, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'x-api-key': API_KEY
                    },
                    body: JSON.stringify({
                        user_id: 'web-user',
                        conversation_id: conversationId,
                        query: query
                    })
                });
                
                const data = await response.json();
                addMessage(data.response, 'agent');
            } catch (error) {
                addMessage('Error: ' + error.message, 'agent');
            } finally {
                document.getElementById('thinking').style.display = 'none';
            }
        }
        
        function addMessage(text, sender) {
            const chat = document.getElementById('chat');
            const msg = document.createElement('div');
            msg.className = 'message ' + sender;
            msg.textContent = text;
            chat.appendChild(msg);
            chat.scrollTop = chat.scrollHeight;
        }
        
        document.getElementById('query').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') sendQuery();
        });
    </script>
</body>
</html>
```
4. Upload to S3
5. Access via S3 website endpoint

### Phase 13: Monitoring and Alerting

**Purpose**: Set up comprehensive monitoring.

**AWS GUI Steps**:

1. Navigate to CloudWatch → Dashboards
2. Create dashboard: "AI-Agents-Dashboard"
3. Add widgets:
   - Lambda invocations
   - Lambda errors
   - Lambda duration
   - DynamoDB read/write capacity
   - Bedrock API calls
   - Step Functions executions
4. Create alarms:
   - Lambda error rate > 1%
   - Lambda duration > 4 minutes
   - DynamoDB throttling
5. Set up SNS topic for notifications

### Phase 14: Cost Optimization

**Purpose**: Optimize agent system costs.

**AWS GUI Steps**:

1. Enable Cost Explorer
2. Create budget: "AI-Agents-Monthly-Budget"
3. Set alerts at 80% and 100%
4. Implement caching in Lambda:
```python
import functools
from datetime import datetime, timedelta

cache = {}

@functools.lru_cache(maxsize=100)
def cached_agent_response(query_hash):
    # Cache responses for identical queries
    pass
```
5. Use Lambda reserved concurrency to prevent runaway costs
6. Monitor Bedrock token usage
7. Implement request throttling in API Gateway

### Phase 15: Security Hardening

**Purpose**: Implement comprehensive security.

**AWS GUI Steps**:

1. Store credentials in Secrets Manager
2. Enable VPC endpoints for Bedrock, DynamoDB
3. Configure AWS WAF for API Gateway
4. Enable CloudTrail for audit logs
5. Implement least-privilege IAM policies
6. Enable encryption for all data at rest
7. Use TLS 1.2+ for all communications

## Production Deployment Checklist

- [ ] All AWS resources created and configured
- [ ] IAM roles follow least-privilege principle
- [ ] Secrets stored in Secrets Manager
- [ ] X-Ray tracing enabled
- [ ] CloudWatch dashboards and alarms configured
- [ ] Bedrock Guardrails implemented
- [ ] Agent memory persisted in DynamoDB
- [ ] Lambda functions tested and deployed
- [ ] API Gateway secured with authentication
- [ ] Step Functions workflows tested
- [ ] Web UI deployed and functional
- [ ] Cost budgets and alerts configured
- [ ] Security scanning completed
- [ ] Backup and recovery procedures documented
- [ ] Load testing completed
- [ ] Documentation updated

## Troubleshooting Guide

### Agent Not Responding
- Check Lambda logs in CloudWatch
- Verify Bedrock model access
- Check DynamoDB table permissions
- Review X-Ray traces for bottlenecks

### Tool Calling Failures
- Verify tool schema format
- Check tool function implementation
- Review Bedrock API response
- Ensure proper error handling

### High Costs
- Review Bedrock token usage
- Implement response caching
- Optimize agent prompts
- Use cheaper models where appropriate

### Memory Issues
- Check DynamoDB item sizes
- Implement conversation pruning
- Review memory retention policy
- Monitor DynamoDB capacity

## Additional Resources

- [AWS Bedrock Tool Use Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use.html)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [Agent Workshop Repository](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker)

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-01  
**Status**: Production Ready
