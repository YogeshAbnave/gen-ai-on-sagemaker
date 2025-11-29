# Interview Materials: Advanced Multi-Modal RAG Application

## Project Introduction Story

"I developed an enterprise-grade, advanced multi-modal RAG (Retrieval-Augmented Generation) application that enables intelligent querying across diverse media types including documents, images, videos, and audio files. The system leverages AWS Bedrock's cutting-edge AI capabilities, including Amazon Nova Sonic for real-time speech-to-speech conversations and Bedrock Data Automation for comprehensive multimedia processing.

The architecture is built on AWS CDK using TypeScript for infrastructure as code, with a React frontend and Python backend services. The application processes uploaded media files through an automated pipeline: files are ingested via S3, processed using Bedrock Data Automation to extract text, transcripts, summaries, and metadata, then indexed in OpenSearch Serverless for vector-based semantic search. Users can interact through both a traditional chat interface and an innovative speech-to-speech interface powered by Amazon Nova Sonic.

Key technical achievements include implementing bidirectional streaming WebSocket connections for real-time voice conversations, integrating Cognito for secure authentication with JWT validation, orchestrating complex event-driven workflows using EventBridge, and designing a scalable serverless architecture using Lambda, ECS Fargate, and CloudFront. The system supports cross-region knowledge base access, implements comprehensive security measures including WAF tagging and SSL enforcement, and provides timestamp-based navigation for video and audio content.

This project demonstrates my expertise in cloud architecture, AI/ML integration, real-time communication protocols, and building production-ready applications that handle complex multimedia processing at scale."

---

## 50+ Technical Interview Questions

### Architecture & Design (10 questions)

1. **Can you walk me through the overall architecture of your multi-modal RAG application?**
   
   **Answer:** The architecture follows a modular, serverless design using AWS CDK with nested stacks. At the core, we have a main MultimediaRagStack that orchestrates five specialized nested stacks:
   
   - **StorageDistStack**: Manages four S3 buckets (media uploads, organized content, multimodal data, application hosting) and a CloudFront distribution with Origin Access Control for secure content delivery. It also handles access logging and lifecycle policies.
   
   - **AuthStack**: Implements authentication using Cognito User Pools and Identity Pools, providing JWT-based authentication and temporary AWS credentials for authenticated users.
   
   - **OpenSearchStack**: Deploys OpenSearch Serverless for vector-based semantic search, configured with data access policies and encryption policies.
   
   - **ProcessingStack**: Contains the processing pipeline with Lambda functions for initial file processing, BDA (Bedrock Data Automation) processing, and retrieval. It also manages the Bedrock Knowledge Base and Data Source.
   
   - **SpeechToSpeechStack**: Runs an ECS Fargate service with a Python WebSocket server for real-time speech-to-speech conversations using Amazon Nova Sonic, fronted by a Network Load Balancer.
   
   The data flow works as follows: Users upload files to S3, which triggers EventBridge notifications. The Initial Processing Lambda routes files based on type - media files go through Bedrock Data Automation for extraction, while documents are moved directly. Processed content is indexed in the Knowledge Base backed by OpenSearch. Users query through either a React chat interface or speech-to-speech interface, both authenticated via Cognito. CloudFront serves the frontend and routes WebSocket traffic to the NLB.

2. **Why did you choose a nested stack architecture in AWS CDK?**
   
   **Answer:** I chose nested stacks for several strategic reasons:
   
   - **Modularity**: Each stack has a single responsibility (storage, auth, processing, etc.), making the codebase easier to understand and maintain. Teams can work on different stacks independently.
   
   - **Resource Limit Management**: CloudFormation has a 500-resource limit per stack. By splitting into nested stacks, we stay well under this limit while building a complex application.
   
   - **Reusability**: Stacks like AuthStack or StorageDistStack can be reused across different projects with minimal modifications.
   
   - **Independent Lifecycle**: We can update the ProcessingStack without affecting the AuthStack, reducing deployment risk and downtime.
   
   - **Avoiding Circular Dependencies**: By carefully ordering stack creation and using exports/imports, we avoid circular dependency issues that would occur in a monolithic stack.
   
   - **Logical Grouping**: Related resources are grouped together, making it easier to apply consistent tagging, security policies, and cost allocation.
   
   The trade-off is slightly more complex deployment orchestration, but CDK handles this automatically through dependency resolution.

3. **How does your application handle the processing pipeline from file upload to searchable content?**
   
   **Answer:** The processing pipeline is event-driven and fully automated:
   
   **Step 1 - Upload & Detection**: When a user uploads a file to the media bucket, S3 emits an EventBridge notification with the object creation event.
   
   **Step 2 - Initial Processing**: The Initial Processing Lambda is triggered by EventBridge. It examines the file extension to determine the type:
   - For media files (mp3, mp4, wav, etc.): Invokes Bedrock Data Automation asynchronously
   - For documents (pdf, txt, etc.): Copies directly to the organized bucket under the Documents/ prefix
   
   **Step 3 - BDA Processing**: For media files, Bedrock Data Automation extracts:
   - Transcripts from audio/video
   - Text from images and documents
   - Summaries and metadata
   - Bounding boxes and timestamps
   
   When BDA completes, it emits an EventBridge event (success or failure).
   
   **Step 4 - Post-Processing**: The BDA Processing Lambda is triggered by the completion event. It:
   - Retrieves the processed output from S3
   - Chunks large files (60-second segments for videos)
   - Formats metadata
   - Stores organized content in the organized bucket
   
   **Step 5 - Knowledge Base Ingestion**: The Bedrock Knowledge Base has a data source configured to monitor the organized bucket. It automatically:
   - Detects new files
   - Generates embeddings using the configured embedding model
   - Indexes vectors in OpenSearch Serverless
   - Makes content searchable
   
   **Step 6 - Retrieval**: When users query, the Retrieval Lambda uses the Knowledge Base Retrieve and Generate API to find relevant content and generate contextual answers.
   
   The entire pipeline is asynchronous, scalable, and handles failures gracefully with retry logic and error notifications.

4. **What design patterns did you implement in this project?**
   
   **Answer:** I implemented several design patterns:
   
   - **Composition Pattern**: The main MultimediaRagStack composes multiple nested stacks, each handling specific concerns. This follows the "favor composition over inheritance" principle.
   
   - **Factory Pattern**: The S2sEvent helper class acts as a factory for creating standardized WebSocket event objects (sessionStart, promptStart, audioInput, etc.), ensuring consistency and reducing errors.
   
   - **Observer Pattern**: EventBridge implements the observer pattern - S3 buckets and BDA services publish events, and Lambda functions subscribe to specific event patterns.
   
   - **Queue Pattern**: Audio processing uses queue-based architecture - the audio_input_queue buffers incoming audio chunks, and the output_queue buffers responses, enabling smooth asynchronous processing.
   
   - **Strategy Pattern**: The Initial Processing Lambda uses different strategies based on file type - BDA strategy for media, direct copy strategy for documents.
   
   - **Singleton Pattern**: The S2sSessionManager maintains a single Bedrock client instance per session, refreshing credentials as needed.
   
   - **Builder Pattern**: CDK constructs use the builder pattern extensively for configuring resources with fluent APIs.
   
   - **Facade Pattern**: The Knowledge Base API provides a simplified facade over the complex operations of embedding generation, vector search, and response generation.

5. **How did you ensure scalability in your architecture?**
   
   **Answer:** Scalability is built into every layer:
   
   - **Serverless Compute**: Lambda functions scale automatically from zero to thousands of concurrent executions. No server management or capacity planning required.
   
   - **Managed Services**: OpenSearch Serverless, Bedrock, and Cognito are fully managed and scale automatically based on demand.
   
   - **ECS Fargate Auto-Scaling**: The speech-to-speech service can scale from 1 to N tasks based on CPU/memory metrics or custom metrics like active WebSocket connections.
   
   - **CloudFront CDN**: Distributes static content globally with edge caching, reducing origin load and improving response times worldwide.
   
   - **Asynchronous Processing**: BDA processing is asynchronous, preventing bottlenecks. Large files don't block the pipeline.
   
   - **Event-Driven Architecture**: EventBridge decouples producers and consumers, allowing independent scaling of each component.
   
   - **S3 Scalability**: S3 automatically scales to handle any number of requests and storage volume.
   
   - **Connection Pooling**: Lambda functions reuse connections and credentials across invocations within the same execution environment.
   
   - **Stateless Design**: All components are stateless, storing state in S3, DynamoDB (if needed), or client-side, enabling horizontal scaling.
   
   - **Chunking Strategy**: Large videos are chunked into 60-second segments, preventing memory issues and enabling parallel processing.
   
   The architecture can handle everything from a single user to thousands of concurrent users without code changes.

6. **Explain your approach to separating concerns between storage, processing, and distribution.**
   
   **Answer:** I applied the Single Responsibility Principle at the stack level:
   
   **StorageDistStack** - Responsible for data persistence and delivery:
   - Creates and configures all S3 buckets with appropriate policies
   - Sets up CloudFront distribution with multiple origins
   - Manages Origin Access Control for secure S3 access
   - Configures access logging and lifecycle rules
   - Handles both storage concerns (buckets) and distribution concerns (CloudFront) because they're tightly coupled
   
   **ProcessingStack** - Handles all data transformation:
   - Lambda functions for file processing
   - Bedrock Data Automation project and processing
   - Knowledge Base and Data Source configuration
   - Retrieval logic
   - Completely independent of how data is stored or delivered
   
   **AuthStack** - Manages identity and access:
   - Cognito User Pool and Identity Pool
   - IAM roles for authenticated users
   - Admin user creation
   - Authentication policies
   - Isolated from storage and processing concerns
   
   **SpeechToSpeechStack** - Real-time communication:
   - ECS Fargate service
   - Network Load Balancer
   - WebSocket server
   - Separate from the main processing pipeline
   
   **Communication Between Stacks**:
   - Stacks communicate through CloudFormation exports/imports
   - The main stack passes bucket references, function ARNs, and IDs between nested stacks
   - No direct coupling - stacks only depend on interfaces (bucket names, ARNs), not implementations
   
   **Benefits**:
   - Can update processing logic without touching storage
   - Can swap authentication providers without affecting processing
   - Can deploy stacks independently (with dependency ordering)
   - Clear ownership boundaries for team collaboration
   - Easier testing - can mock stack interfaces

7. **How does your application handle different media types (documents, images, videos, audio)?**
   
   **Answer:** The application uses a type-detection and routing strategy:
   
   **Detection Phase** (Initial Processing Lambda):
   ```python
   file_extension = os.path.splitext(key)[1].lower()
   MEDIA_EXTENSIONS = ['.mp3', '.mp4', '.wav', '.flac', '.ogg', '.amr', '.webm']
   
   if file_extension in MEDIA_EXTENSIONS:
       # Route to BDA for media processing
   else:
       # Route to direct document processing
   ```
   
   **Bedrock Data Automation Configuration**:
   Each media type has specific extraction settings:
   
   - **Documents**: Extract text at PAGE, ELEMENT, and WORD granularity with bounding boxes. Enable document splitting for large files.
   
   - **Images**: Extract text via OCR, detect content moderation issues, generate image summaries, and extract IAB (Interactive Advertising Bureau) categories.
   
   - **Videos**: Extract transcripts with timestamps, detect text in frames, identify content moderation issues, generate video summaries, chapter summaries, and IAB categories.
   
   - **Audio**: Extract transcripts, detect audio content moderation issues, generate audio summaries and topic summaries.
   
   **Post-Processing**:
   The BDA Processing Lambda handles type-specific logic:
   - Videos are chunked into 60-second segments for better retrieval granularity
   - Timestamps are preserved for video/audio navigation
   - Metadata is formatted consistently across types
   
   **Frontend Rendering**:
   - Documents: Display with "Know More" buttons linking to original files
   - Videos/Audio: Parse timestamps and render VideoPopover components for in-place playback
   - Images: Display inline with extracted text
   
   **Knowledge Base Indexing**:
   All processed content is indexed uniformly in OpenSearch, but metadata preserves the original type for filtering and specialized retrieval.

8. **What considerations did you make for cross-region deployments?**
   
   **Answer:** Cross-region support was critical due to service availability constraints:
   
   **Primary Challenge**: Amazon Nova Sonic (speech-to-speech) is only available in us-east-1, but the main application might be deployed in other regions.
   
   **Solutions Implemented**:
   
   1. **Regional Service Deployment**:
      - SpeechToSpeechStack explicitly sets AWS_DEFAULT_REGION to 'us-east-1'
      - The ECS task always runs in us-east-1 regardless of main stack region
   
   2. **Cross-Region Knowledge Base Access**:
      - Knowledge Base might be in us-west-2 while speech service is in us-east-1
      - Environment variables configure both KB_REGION and BACKEND_REGION
      - IAM policies use wildcards for cross-region Bedrock Agent Runtime access
   
   3. **Credential Management**:
      - Boto3 sessions are region-aware
      - Credentials are refreshed with proper region context
      - The S2sSessionManager handles cross-region API calls transparently
   
   4. **Configuration Flexibility**:
      ```typescript
      kbRegion?: string;  // Knowledge Base region
      ragModelArn?: string;  // Model ARN (region-specific)
      ```
   
   5. **Network Considerations**:
      - CloudFront provides global edge locations regardless of origin region
      - WebSocket connections route through CloudFront to the us-east-1 NLB
      - Latency is minimized through CloudFront's global network
   
   6. **Data Residency**:
      - S3 buckets remain in the deployment region for data sovereignty
      - Only real-time speech processing happens in us-east-1
      - Processed data returns to the original region
   
   7. **Monitoring**:
      - CloudWatch metrics are region-specific
      - Cross-region dashboards aggregate metrics from multiple regions
   
   **Future Considerations**:
   - When Nova Sonic becomes available in more regions, we can deploy regionally
   - The architecture supports multi-region active-active deployment
   - Route 53 could provide region-based routing

9. **How did you design the WebSocket architecture for speech-to-speech functionality?**
   
   **Answer:** The WebSocket architecture handles real-time bidirectional audio streaming:
   
   **Server Architecture**:
   - Python `websockets` library for WebSocket server
   - Asyncio for concurrent connection handling
   - Separate health check HTTP server on port 8082
   - Main WebSocket server on port 8081
   
   **Connection Lifecycle**:
   
   1. **Authentication**: 
      - Client includes JWT token in query parameter: `wss://domain/ws/speech-to-speech?token=xxx`
      - Server validates token against Cognito JWKS before accepting connection
      - Invalid tokens result in immediate connection closure with code 1008
   
   2. **Session Initialization**:
      - Create S2sSessionManager instance per connection
      - Initialize bidirectional stream with Bedrock
      - Start audio processing task and response forwarding task
   
   3. **Audio Flow**:
      - **Client → Server**: Microphone captures audio → resample to 16kHz → PCM encode → base64 → WebSocket
      - **Server → Bedrock**: Audio chunks queued → sent to Bedrock bidirectional stream
      - **Bedrock → Server**: Responses queued in output_queue
      - **Server → Client**: Responses forwarded via WebSocket
   
   4. **Event-Based Communication**:
      ```python
      # Client sends structured events
      {
        "event": {
          "sessionStart": {},
          "promptStart": {...},
          "audioInput": {...},
          "contentEnd": {...}
        }
      }
      ```
   
   5. **Connection Management**:
      - Ping/pong every 20 seconds for keepalive
      - 30-second pong timeout
      - Graceful shutdown on session end
      - Automatic cleanup of resources
   
   6. **Error Handling**:
      - Connection errors logged with connection ID
      - Credential refresh on auth errors
      - Automatic reconnection logic on client side
   
   7. **Scalability**:
      - Each connection runs in separate asyncio task
      - ECS Fargate can scale to multiple tasks
      - NLB distributes connections across tasks
      - Stateless design allows horizontal scaling
   
   **Load Balancing**:
   - Network Load Balancer for TCP/WebSocket traffic
   - Health checks on dedicated port
   - Cross-zone load balancing enabled
   - Connection draining during deployments

10. **What trade-offs did you consider when choosing between synchronous and asynchronous processing?**
    
    **Answer:** I carefully evaluated sync vs async for each component:
    
    **Asynchronous Processing (Chosen for)**:
    
    - **BDA Processing**: 
      - Trade-off: Complexity of event handling vs. ability to process large files
      - Decision: Async - Videos can take minutes to process, would timeout Lambda
      - Benefit: No timeout constraints, better resource utilization
    
    - **Knowledge Base Ingestion**:
      - Trade-off: Immediate availability vs. system simplicity
      - Decision: Async - Automatic ingestion on S3 changes
      - Benefit: Decoupled from upload flow, handles batch updates efficiently
    
    - **Audio Queue Processing**:
      - Trade-off: Latency vs. reliability
      - Decision: Async queue - Buffers audio chunks
      - Benefit: Smooth streaming even with network jitter
    
    **Synchronous Processing (Chosen for)**:
    
    - **Retrieval Lambda**:
      - Trade-off: User wait time vs. implementation simplicity
      - Decision: Sync - Users expect immediate chat responses
      - Benefit: Simple request-response model, easier error handling
    
    - **JWT Validation**:
      - Trade-off: Connection latency vs. security
      - Decision: Sync - Must validate before accepting connection
      - Benefit: Immediate rejection of invalid tokens
    
    - **Initial File Routing**:
      - Trade-off: Upload confirmation vs. processing time
      - Decision: Sync for routing decision, async for actual processing
      - Benefit: Fast upload confirmation, heavy processing doesn't block
    
    **Hybrid Approach**:
    
    - **WebSocket Communication**:
      - Bidirectional async streams for audio
      - Sync event acknowledgment
      - Best of both worlds: real-time feel with async scalability
    
    **Key Considerations**:
    - User experience: Sync for user-facing operations under 3 seconds
    - Resource limits: Async for operations that might exceed Lambda timeout
    - Cost: Async reduces idle time and costs
    - Complexity: Sync is simpler but less scalable
    - Error handling: Async requires more sophisticated retry and dead-letter queue strategies

### AWS Services & Cloud (12 questions)

11. **How did you implement authentication and authorization in your application?**
    - Expected: Cognito User Pools, Identity Pools, JWT token validation, IAM roles for authenticated users, Lambda@Edge for CloudFront.

12. **Explain your use of AWS Bedrock in this project.**
    - Expected: Bedrock Data Automation for multimedia processing, Knowledge Bases for RAG, Nova Sonic for speech-to-speech, model invocation APIs.

13. **How does Bedrock Data Automation work in your pipeline?**
    - Expected: Async invocation, EventBridge notifications, extraction configurations (granularity, bounding boxes), output formats.

14. **What is your strategy for managing S3 buckets and their lifecycle?**
    - Expected: Separate buckets for media/organized/multimodal/logs, lifecycle rules for cost optimization, encryption, access logging.

15. **How did you implement the Knowledge Base integration?**
    - Expected: OpenSearch Serverless as vector store, data source configuration, embedding models, retrieval and generate APIs.

16. **Explain your CloudFront distribution setup.**
    - Expected: Origin Access Control for S3, multiple origins (S3, NLB), behavior patterns, caching policies, Lambda@Edge integration.

17. **How do you handle Lambda cold starts and performance optimization?**
    - Expected: Dependency layers, memory allocation, timeout configuration, connection pooling, credential caching.

18. **What role does EventBridge play in your architecture?**
    - Expected: S3 event notifications, BDA completion events, decoupling services, event pattern matching, multiple targets.

19. **How did you implement the ECS Fargate service for speech backend?**
    - Expected: Task definitions, ECR for container images, Network Load Balancer, health checks, auto-scaling, VPC configuration.

20. **Explain your approach to IAM roles and permissions.**
    - Expected: Least privilege principle, service-specific roles, inline vs managed policies, cross-service permissions, trust relationships.

21. **How do you handle secrets and environment variables?**
    - Expected: SSM Parameter Store for Cognito config, environment variables in Lambda/ECS, credential rotation, secure transmission.

22. **What monitoring and logging strategies did you implement?**
    - Expected: CloudWatch Logs for Lambda/ECS, access logs for S3/CloudFront, custom metrics, dashboards, log retention policies.

### Backend Development (10 questions)

23. **Walk me through your WebSocket server implementation.**
    - Expected: Python websockets library, connection handling, authentication via query params, message routing, graceful shutdown.

24. **How does the S2sSessionManager class work?**
    - Expected: Bidirectional streaming with Bedrock, audio queue management, credential refresh, event processing, tool use handling.

25. **Explain your audio processing pipeline in the speech-to-speech feature.**
    - Expected: Microphone capture, resampling to 16kHz, PCM encoding, base64 encoding, chunking, queue-based sending.

26. **How do you handle JWT token validation in the WebSocket server?**
    - Expected: JWKS fetching and caching, RSA signature verification, expiration checks, user extraction, error handling.

27. **What is your strategy for managing WebSocket connection lifecycle?**
    - Expected: Connection tracking, session management, cleanup on disconnect, ping/pong for keepalive, error recovery.

28. **How does the tool use functionality work in your speech-to-speech system?**
    - Expected: Tool detection in responses, knowledge base queries, result formatting, event sequencing (start/result/end).

29. **Explain your approach to credential management in the Python backend.**
    - Expected: Boto3 session management, automatic refresh, expiration tracking, environment variable updates, error handling.

30. **How do you handle audio playback synchronization in the frontend?**
    - Expected: Audio queue, sequential playback, promise-based loading, interruption handling, state management.

31. **What error handling strategies did you implement in the Lambda functions?**
    - Expected: Try-catch blocks, logging, graceful degradation, retry logic, error responses, CloudWatch integration.

32. **How does the BDA processing Lambda handle different output formats?**
    - Expected: JSON parsing, S3 key generation, metadata extraction, chunking for large files, error scenarios.

### Frontend Development (8 questions)

33. **Describe your React application architecture.**
    - Expected: Component structure, context API for state, AWS Amplify integration, custom hooks, routing.

34. **How did you implement the chat interface with multimedia support?**
    - Expected: Message parsing, timestamp extraction, video popover component, metadata display, dynamic rendering.

35. **Explain your approach to managing AWS SDK calls from the frontend.**
    - Expected: Credential provider from Cognito, Lambda invocation, S3 uploads, error handling, loading states.

36. **How does the speech-to-speech component manage real-time audio?**
    - Expected: WebSocket connection, microphone access, audio context, resampling, event handling, state management.

37. **What accessibility considerations did you implement?**
    - Expected: ARIA labels, keyboard navigation, screen reader support, CloudScape components, semantic HTML.

38. **How do you handle file uploads in your application?**
    - Expected: S3 multipart upload, progress tracking, file validation, error handling, signed URLs.

39. **Explain your state management strategy.**
    - Expected: React hooks (useState, useEffect, useRef), Context API for global state, local vs global state decisions.

40. **How did you implement the video timestamp navigation feature?**
    - Expected: Timestamp parsing from responses, VideoPopover component, player control, URL generation, CloudFront integration.

### Infrastructure as Code (6 questions)

41. **Why did you choose AWS CDK over CloudFormation or Terraform?**
    - Expected: Type safety, reusability, programming language familiarity, L2/L3 constructs, testing capabilities.

42. **How do you manage CDK stack dependencies?**
    - Expected: Explicit dependencies, cross-stack references, exports/imports, nested stacks, deployment order.

43. **Explain your approach to environment-specific configurations.**
    - Expected: Resource suffix pattern, environment variables, conditional deployments, parameter passing.

44. **How do you handle CDK custom resources?**
    - Expected: Lambda-backed custom resources, BDA project creation, layer creation, lifecycle management, error handling.

45. **What testing strategies do you use for CDK code?**
    - Expected: Unit tests, snapshot tests, integration tests, fine-grained assertions, mocking.

46. **How do you manage CDK deployment and updates?**
    - Expected: Bootstrap process, deployment scripts, change sets, rollback strategies, blue-green deployments.

### Security & Best Practices (8 questions)

47. **What security measures did you implement in your application?**
    - Expected: Cognito authentication, JWT validation, IAM least privilege, SSL enforcement, bucket policies, WAF tagging.

48. **How do you handle CORS in your application?**
    - Expected: S3 CORS configuration, CloudFront headers, API Gateway CORS, preflight requests.

49. **Explain your approach to securing S3 buckets.**
    - Expected: Block public access, encryption at rest, SSL enforcement, access logging, bucket policies, OAC.

50. **How do you manage sensitive data and PII?**
    - Expected: Encryption, access controls, logging exclusions, data retention policies, compliance considerations.

51. **What logging and auditing capabilities did you implement?**
    - Expected: CloudWatch Logs, S3 access logs, CloudFront logs, structured logging, log aggregation, retention.

52. **How do you ensure high availability and disaster recovery?**
    - Expected: Multi-AZ deployments, serverless resilience, backup strategies, health checks, failover mechanisms.

53. **What cost optimization strategies did you implement?**
    - Expected: S3 lifecycle policies, Lambda memory optimization, CloudFront caching, serverless architecture, resource tagging.

54. **How do you handle rate limiting and throttling?**
    - Expected: Lambda concurrency limits, API throttling, exponential backoff, queue-based processing, error handling.

### Problem-Solving & Debugging (6 questions)

55. **Describe a challenging bug you encountered and how you resolved it.**
    - Expected: Specific example with problem description, debugging approach, root cause analysis, solution, prevention.

56. **How do you debug WebSocket connection issues?**
    - Expected: Browser dev tools, server logs, connection state tracking, ping/pong monitoring, network analysis.

57. **What tools do you use for monitoring and troubleshooting AWS services?**
    - Expected: CloudWatch, X-Ray, CloudTrail, AWS CLI, SDK debugging, log insights, metrics and alarms.

58. **How do you handle Lambda timeout issues?**
    - Expected: Timeout configuration, async processing, step functions, progress tracking, partial results.

59. **Describe your approach to performance optimization.**
    - Expected: Profiling, bottleneck identification, caching strategies, lazy loading, code splitting, CDN usage.

60. **How do you ensure code quality and maintainability?**
    - Expected: Code reviews, linting, formatting, documentation, testing, modular design, naming conventions.

---

## Additional Behavioral Questions

61. How did you prioritize features during development?
62. Describe your collaboration process with team members.
63. How do you stay updated with AWS services and best practices?
64. What would you improve if you had more time?
65. How do you handle technical debt in your projects?

---

## Key Technical Highlights to Emphasize

- **Real-time bidirectional streaming** with WebSocket and Bedrock
- **Multi-modal AI processing** across documents, images, video, and audio
- **Serverless architecture** for scalability and cost efficiency
- **Infrastructure as Code** with TypeScript CDK
- **Event-driven design** using EventBridge
- **Secure authentication** with Cognito and JWT
- **Cross-region capabilities** for knowledge base access
- **Production-ready** with logging, monitoring, and error handling


### AWS Services & Cloud - Detailed Answers

11. **How did you implement authentication and authorization in your application?**
    
    **Answer:** I implemented a comprehensive auth system using AWS Cognito with multiple layers:
    
    **User Authentication (Cognito User Pool)**:
    - Self-signup enabled with email verification
    - Password policy: 8+ characters, uppercase, lowercase, digits
    - Email-based sign-in
    - Account recovery via email
    - Admin group for privileged users
    
    **Application Integration (User Pool Client)**:
    - Multiple auth flows: SRP, user password, admin user password
    - OAuth 2.0 flows: implicit and authorization code grant
    - Token validity: 5 minutes for access/ID tokens, 7 days for refresh tokens
    - Scopes: profile, email, openid, phone, cognito admin
    
    **AWS Resource Access (Identity Pool)**:
    - Federated identities from User Pool
    - Temporary AWS credentials via STS
    - Server-side token verification enabled
    - Separate roles for authenticated and unauthenticated users
    
    **Authorization (IAM Roles)**:
    ```typescript
    authenticatedRole.addToPolicy(
      new PolicyStatement({
        actions: ['lambda:InvokeFunction'],
        resources: [retrievalFunction.functionArn]
      })
    );
    ```
    - Authenticated users can: invoke retrieval Lambda, access S3 media bucket, start Bedrock ingestion jobs
    - Least privilege principle applied
    
    **WebSocket Authentication**:
    - JWT token passed as query parameter
    - Server-side validation against Cognito JWKS
    - RSA signature verification
    - Expiration and issuer checks
    - Connection rejected if invalid
    
    **CloudFront Protection (Optional Lambda@Edge)**:
    - JWT validation at edge locations
    - Query parameter-based auth
    - SSM Parameter Store for Cognito config
    - Reduces load on origin servers
    
    **Security Features**:
    - SSL/TLS enforcement everywhere
    - Token refresh mechanism
    - Automatic credential rotation
    - Audit logging via CloudTrail

12. **Explain your use of AWS Bedrock in this project.**
    
    **Answer:** Bedrock is central to the AI capabilities, used in three distinct ways:
    
    **1. Bedrock Data Automation (BDA)**:
    - Processes multimedia files (documents, images, videos, audio)
    - Extracts structured data: text, transcripts, summaries, metadata
    - Configuration per media type:
      ```python
      "video": {
        "extraction": {
          "category": {
            "types": ["TRANSCRIPT", "TEXT_DETECTION", "CONTENT_MODERATION"]
          }
        },
        "generativeField": {
          "types": ["VIDEO_SUMMARY", "CHAPTER_SUMMARY", "IAB"]
        }
      }
      ```
    - Async invocation with EventBridge notifications
    - Handles large files without timeout issues
    
    **2. Knowledge Bases for RAG**:
    - Stores processed content with vector embeddings
    - Uses Amazon Titan Embeddings model (configurable)
    - OpenSearch Serverless as vector store
    - Data source monitors S3 for automatic ingestion
    - Retrieve and Generate API for contextual responses
    - Combines semantic search with LLM generation
    
    **3. Amazon Nova Sonic (Speech-to-Speech)**:
    - Real-time bidirectional streaming
    - Voice-to-voice conversations
    - Supports multiple voices (Matthew, Tiffany, Amy)
    - Tool use capability for knowledge base queries
    - Interruption detection and handling
    - Low-latency audio processing
    
    **4. Foundation Models**:
    - Claude 3 Haiku for RAG responses (configurable)
    - Titan Embeddings for vector generation
    - Model invocation via InvokeModel API
    - Streaming responses for better UX
    
    **Integration Benefits**:
    - Fully managed AI services
    - No model training required
    - Automatic scaling
    - Pay-per-use pricing
    - Enterprise-grade security and compliance

13. **How does Bedrock Data Automation work in your pipeline?**
    
    **Answer:** BDA is a sophisticated async processing system:
    
    **Invocation Phase**:
    ```python
    response = bedrock_data_automation.invoke_data_automation_async(
        inputConfiguration={'s3Uri': f's3://{source_bucket}/{key}'},
        outputConfiguration={'s3Uri': f's3://{target_bucket}/bda-output/{filename}/'},
        dataAutomationConfiguration={
            'dataAutomationProjectArn': project_arn,
            'stage': 'LIVE'
        },
        notificationConfiguration={
            'eventBridgeConfiguration': {'eventBridgeEnabled': True}
        }
    )
    ```
    
    **Project Configuration**:
    - Created via custom resource Lambda
    - Defines extraction settings per media type
    - Granularity levels: PAGE, ELEMENT, WORD
    - Bounding box extraction enabled
    - Generative fields for summaries
    - Document splitter for large files
    
    **Processing Flow**:
    1. Initial Lambda invokes BDA async
    2. BDA processes file (can take minutes for videos)
    3. Extracts text, transcripts, metadata
    4. Generates summaries and classifications
    5. Writes structured output to S3
    6. Emits EventBridge event on completion
    
    **Event Handling**:
    ```python
    eventPattern: {
      source: ['aws.bedrock'],
      detailType: [
        'Bedrock Data Automation Job Succeeded',
        'Bedrock Data Automation Job Failed With Client Error',
        'Bedrock Data Automation Job Failed With Service Error'
      ]
    }
    ```
    
    **Output Structure**:
    - JSON files with extracted data
    - Separate files for different granularities
    - Metadata includes timestamps, confidence scores
    - Additional files for summaries and classifications
    
    **Error Handling**:
    - Client errors: Invalid input format, unsupported file type
    - Service errors: Temporary failures, retry automatically
    - Failed jobs logged to CloudWatch
    - Dead letter queue for persistent failures
    
    **Benefits**:
    - No timeout constraints
    - Handles files of any size
    - Parallel processing of multiple files
    - Consistent output format
    - Built-in retry logic

14. **What is your strategy for managing S3 buckets and their lifecycle?**
    
    **Answer:** I implemented a comprehensive S3 management strategy:
    
    **Bucket Architecture**:
    
    1. **Media Bucket** (User Uploads):
       - EventBridge notifications enabled
       - CORS configured for browser uploads
       - Server access logging to log bucket
       - Encryption: S3-managed (SSE-S3)
       - Block all public access
       - SSL enforcement via bucket policy
    
    2. **Organized Bucket** (Processed Content):
       - EventBridge enabled for Knowledge Base
       - Lifecycle rule: transition to IA after 30 days
       - Prefix-based organization: Documents/, bda-output/, raw-transcripts/
       - Access logging enabled
       - Encryption at rest
    
    3. **Multimodal Bucket** (Knowledge Base Data):
       - Dedicated for Knowledge Base operations
       - Strict access controls
       - Encryption enabled
       - Access logging
    
    4. **Application Host Bucket** (React Frontend):
       - CloudFront Origin Access Control
       - No public access
       - Removal policy: DESTROY (dev environment)
       - Auto-delete objects on stack deletion
    
    5. **Access Logs Bucket**:
       - Centralized logging for all buckets
       - Versioning enabled
       - Lifecycle rules:
         - 30 days → Infrequent Access
         - 90 days → Glacier
         - 365 days → Delete
       - Object ownership: BUCKET_OWNER_PREFERRED
    
    **Security Measures**:
    ```typescript
    bucket.addToResourcePolicy(
      new PolicyStatement({
        actions: ['s3:GetObject'],
        principals: [new ServicePrincipal('cloudfront.amazonaws.com')],
        resources: [bucket.arnForObjects('*')],
        conditions: {
          StringEquals: {
            'AWS:SourceArn': `arn:aws:cloudfront::${accountId}:distribution/${distributionId}`
          }
        }
      })
    );
    ```
    
    **Cost Optimization**:
    - Lifecycle transitions to cheaper storage classes
    - Automatic deletion of old logs
    - Intelligent-Tiering for unpredictable access patterns
    - Multipart upload for large files
    
    **Monitoring**:
    - CloudWatch metrics for bucket size and request count
    - S3 Inventory for object-level insights
    - Access logs for security auditing
    - Cost allocation tags

15. **How did you implement the Knowledge Base integration?**
    
    **Answer:** The Knowledge Base integration is sophisticated and fully automated:
    
    **Knowledge Base Creation**:
    ```typescript
    const knowledgeBase = new bedrock.CfnKnowledgeBase(this, 'KnowledgeBase', {
      name: `documents-kb-${resourceSuffix}`,
      roleArn: kbRole.roleArn,
      knowledgeBaseConfiguration: {
        type: 'VECTOR',
        vectorKnowledgeBaseConfiguration: {
          embeddingModelArn: `arn:aws:bedrock:${region}::foundation-model/${embeddingModelId}`
        }
      },
      storageConfiguration: {
        type: 'OPENSEARCH_SERVERLESS',
        opensearchServerlessConfiguration: {
          collectionArn: opensearchCollection.attrArn,
          vectorIndexName: 'bedrock-knowledge-base-index',
          fieldMapping: {
            vectorField: 'bedrock-knowledge-base-vector',
            textField: 'AMAZON_BEDROCK_TEXT_CHUNK',
            metadataField: 'AMAZON_BEDROCK_METADATA'
          }
        }
      }
    });
    ```
    
    **Data Source Configuration**:
    - Monitors organized S3 bucket
    - Automatic ingestion on file changes
    - Chunking strategy for large documents
    - Metadata preservation
    - Incremental updates
    
    **OpenSearch Serverless Setup**:
    - Collection with encryption policy
    - Data access policy for Bedrock role
    - Network policy for VPC access (if needed)
    - Index with vector field configuration
    - Dimension matching embedding model
    
    **IAM Permissions**:
    ```typescript
    kbRole.addToPolicy(new PolicyStatement({
      actions: ['aoss:APIAccessAll'],
      resources: [opensearchCollection.attrArn]
    }));
    
    kbRole.addToPolicy(new PolicyStatement({
      actions: ['s3:GetObject', 's3:ListBucket'],
      resources: [organizedBucket.bucketArn, `${organizedBucket.bucketArn}/*`]
    }));
    
    kbRole.addToPolicy(new PolicyStatement({
      actions: ['bedrock:InvokeModel'],
      resources: [`arn:aws:bedrock:${region}::foundation-model/${embeddingModelId}`]
    }));
    ```
    
    **Retrieval Process**:
    1. User submits query via chat or speech
    2. Retrieval Lambda calls RetrieveAndGenerate API
    3. Query is embedded using Titan model
    4. Vector search in OpenSearch finds relevant chunks
    5. Top results passed to LLM (Claude) with context
    6. LLM generates contextual response
    7. Response includes source citations
    
    **Advanced Features**:
    - Semantic search (not just keyword matching)
    - Metadata filtering (by file type, date, etc.)
    - Relevance scoring
    - Source attribution
    - Multi-document synthesis
    
    **Monitoring**:
    - CloudWatch metrics for query latency
    - Ingestion job status tracking
    - Error logging for failed ingestions
    - Cost tracking per query

(Continuing with remaining AWS questions...)

1
6. **Explain your CloudFront distribution setup.**
    
    **Answer:** CloudFront serves as the global entry point with sophisticated routing:
    
    **Multiple Origins**:
    1. **S3 Origin (Media Bucket)**: Default behavior for uploaded media files
    2. **S3 Origin (App Bucket)**: Static website files (HTML, JS, CSS)
    3. **NLB Origin (Speech Service)**: WebSocket connections for speech-to-speech
    
    **Origin Access Control (OAC)**:
    ```typescript
    const oac = new cloudfront.CfnOriginAccessControl(this, 'OAC', {
      originAccessControlConfig: {
        name: `multimedia-rag-oac-${suffix}`,
        signingBehavior: 'always',
        signingProtocol: 'sigv4',
        originAccessControlOriginType: 's3'
      }
    });
    ```
    - Replaces legacy Origin Access Identity
    - Uses SigV4 signing for S3 requests
    - More secure than OAI
    - Applied to all S3 origins
    
    **Behavior Patterns**:
    ```typescript
    additionalBehaviors: {
      '*.html': {
        origin: appBucketS3Origin,
        cachePolicy: CachePolicy.CACHING_DISABLED,
        compress: true
      },
      '*.js': { /* similar config */ },
      '*.css': { /* similar config */ },
      '/ws/*': {
        origin: nlbOrigin,
        allowedMethods: AllowedMethods.ALLOW_ALL,
        cachePolicy: CachePolicy.CACHING_DISABLED,
        originRequestPolicy: OriginRequestPolicy.ALL_VIEWER
      }
    }
    ```
    
    **Caching Strategy**:
    - Static assets: Disabled for development (would enable with versioning in prod)
    - API responses: Disabled to ensure fresh data
    - WebSocket: No caching (real-time communication)
    - Future: Enable caching with cache invalidation on updates
    
    **Security Configuration**:
    - Minimum TLS version: 1.2 with modern ciphers
    - HTTPS redirect for all HTTP requests
    - CORS headers via response headers policy
    - Security headers (CSP, X-Frame-Options, etc.)
    
    **Lambda@Edge Integration** (Optional):
    ```typescript
    edgeLambdas: [{
      functionVersion: lambda.Version.fromVersionArn(this, 'EdgeFn', edgeLambdaArn),
      eventType: LambdaEdgeEventType.ORIGIN_REQUEST
    }]
    ```
    - JWT validation at edge
    - Reduces origin load
    - Lower latency for auth checks
    
    **Logging**:
    - Access logs to S3
    - Includes cookies for investigation
    - Prefix: cloudfront-logs/
    - Retention via S3 lifecycle
    
    **Performance**:
    - HTTP/2 enabled
    - IPv6 enabled
    - Compression enabled
    - Price class: 100 (North America, Europe)
    - Global edge locations

17. **How do you handle Lambda cold starts and performance optimization?**
    
    **Answer:** I implemented several strategies to minimize cold starts and optimize performance:
    
    **Dependency Layers**:
    ```typescript
    const dependencyLayer = new lambda.LayerVersion(this, 'DependencyLayer', {
      code: lambda.Code.fromBucket(layerBucket, 'layer.zip'),
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_12]
    });
    ```
    - Separates dependencies from function code
    - Reduces deployment package size
    - Shared across multiple functions
    - Cached by Lambda service
    - Faster cold starts
    
    **Memory Allocation**:
    - Retrieval Lambda: 512 MB (balanced for AI workloads)
    - Processing Lambdas: 256 MB (I/O bound)
    - Higher memory = more CPU = faster execution
    - Cost vs performance trade-off
    
    **Timeout Configuration**:
    - Processing functions: 15 minutes (max for async operations)
    - Retrieval function: 15 minutes (Bedrock calls can be slow)
    - Layer creator: 15 minutes (pip install can be slow)
    - Appropriate timeouts prevent premature termination
    
    **Connection Pooling**:
    ```python
    # Reuse clients across invocations
    s3_client = boto3.client('s3')  # Outside handler
    
    def lambda_handler(event, context):
        # Reuses existing connection
        s3_client.get_object(...)
    ```
    
    **Credential Caching**:
    - Boto3 credentials cached in execution environment
    - JWKS cached for 1 hour in WebSocket server
    - Reduces API calls to Cognito
    
    **Provisioned Concurrency** (Not implemented but considered):
    - Would keep functions warm
    - Eliminates cold starts for critical paths
    - Higher cost but better UX
    - Good for production with predictable traffic
    
    **Code Optimization**:
    - Lazy imports (import only when needed)
    - Minimal dependencies
    - Efficient algorithms
    - Avoid heavy initialization in global scope
    
    **Async Processing**:
    - Long-running tasks moved to async (BDA)
    - Reduces Lambda execution time
    - Better resource utilization
    
    **Monitoring**:
    - CloudWatch metrics for cold start duration
    - X-Ray tracing for bottleneck identification
    - Custom metrics for optimization tracking

18. **What role does EventBridge play in your architecture?**
    
    **Answer:** EventBridge is the nervous system of the application, enabling event-driven architecture:
    
    **S3 Event Notifications**:
    ```typescript
    const fileProcessingRule = new events.Rule(this, 'FileProcessingRule', {
      eventPattern: {
        source: ['aws.s3'],
        detailType: ['Object Created'],
        detail: {
          bucket: { name: [mediaBucket.bucketName] }
        }
      }
    });
    fileProcessingRule.addTarget(new targets.LambdaFunction(initialProcessingFunction));
    ```
    - Triggers on file upload to media bucket
    - Replaces legacy S3 notifications
    - More flexible filtering
    - Multiple targets possible
    
    **BDA Completion Events**:
    ```typescript
    const bdaEventRule = new events.Rule(this, 'BDAEventRule', {
      eventPattern: {
        source: ['aws.bedrock', 'aws.bedrock-test'],
        detailType: [
          'Bedrock Data Automation Job Succeeded',
          'Bedrock Data Automation Job Failed With Client Error',
          'Bedrock Data Automation Job Failed With Service Error'
        ]
      }
    });
    ```
    - Captures BDA async job completions
    - Handles both success and failure cases
    - Enables post-processing logic
    
    **Benefits of EventBridge**:
    
    1. **Decoupling**: Producers don't know about consumers
    2. **Scalability**: Handles millions of events
    3. **Filtering**: Complex pattern matching
    4. **Multiple Targets**: One event → many consumers
    5. **Reliability**: Built-in retry and DLQ
    6. **Observability**: Event history and monitoring
    7. **Schema Registry**: Event schema validation
    
    **Event Flow Example**:
    ```
    User uploads video.mp4
    → S3 emits Object Created event
    → EventBridge matches pattern
    → Triggers Initial Processing Lambda
    → Lambda invokes BDA async
    → BDA processes video (minutes)
    → BDA emits Job Succeeded event
    → EventBridge matches pattern
    → Triggers BDA Processing Lambda
    → Lambda organizes output
    → Knowledge Base auto-ingests
    ```
    
    **Error Handling**:
    - Failed events sent to DLQ
    - Retry policy: 3 attempts with exponential backoff
    - CloudWatch alarms on DLQ depth
    - Manual replay capability
    
    **Future Enhancements**:
    - Event archive for compliance
    - Cross-account event routing
    - Custom event bus for application events
    - Event replay for testing

19. **How did you implement the ECS Fargate service for speech backend?**
    
    **Answer:** The ECS Fargate service provides a containerized, scalable WebSocket server:
    
    **VPC Configuration**:
    ```typescript
    const vpc = new ec2.Vpc(this, 'SpeechBackendVpc', {
      maxAzs: 2,
      natGateways: 1
    });
    ```
    - 2 Availability Zones for high availability
    - Public and private subnets
    - NAT Gateway for outbound internet access
    - Security groups for network isolation
    
    **ECR Repository**:
    - Stores Docker images
    - Versioned with tags (latest, v1.0.0, etc.)
    - Scanned for vulnerabilities
    - Lifecycle policy to clean old images
    
    **Task Definition**:
    ```typescript
    const taskDefinition = new ecs.FargateTaskDefinition(this, 'TaskDef', {
      memoryLimitMiB: 2048,
      cpu: 1024,
      executionRole,  // Pulls image, writes logs
      taskRole        // Application permissions
    });
    
    taskDefinition.addContainer('Container', {
      image: ecs.ContainerImage.fromEcrRepository(repository, 'latest'),
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: 'speech-backend' }),
      environment: {
        HOST: '0.0.0.0',
        WS_PORT: '8081',
        HEALTH_PORT: '8082',
        AWS_DEFAULT_REGION: 'us-east-1',
        DOCUMENTS_KB_ID: knowledgeBaseId,
        USE_RAG: 'true'
      },
      portMappings: [
        { containerPort: 8081 },  // WebSocket
        { containerPort: 8082 }   // Health check
      ]
    });
    ```
    
    **IAM Roles**:
    - **Execution Role**: ECS agent permissions (pull image, write logs)
    - **Task Role**: Application permissions (Bedrock, Cognito, CloudWatch)
    
    **Fargate Service**:
    ```typescript
    const service = new ecs.FargateService(this, 'Service', {
      cluster,
      taskDefinition,
      desiredCount: 1,
      assignPublicIp: true,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      minHealthyPercent: 0,
      maxHealthyPercent: 200,
      healthCheckGracePeriod: Duration.seconds(600)
    });
    ```
    - Starts with 1 task (can scale to N)
    - Public IP for NLB access
    - Rolling deployment strategy
    - 10-minute grace period for startup
    
    **Network Load Balancer**:
    ```typescript
    const nlb = new NetworkLoadBalancer(this, 'NLB', {
      vpc,
      internetFacing: true,
      crossZoneEnabled: true
    });
    
    const listener = nlb.addListener('Listener', {
      port: 8081,
      protocol: Protocol.TCP
    });
    
    listener.addTargets('Targets', {
      port: 8081,
      targets: [service],
      healthCheck: {
        port: '8082',
        protocol: Protocol.HTTP,
        path: '/health',
        interval: Duration.seconds(30),
        timeout: Duration.seconds(10),
        healthyThresholdCount: 2,
        unhealthyThresholdCount: 10
      }
    });
    ```
    
    **Health Checks**:
    - Dedicated HTTP endpoint on port 8082
    - Returns JSON: `{"status": "healthy"}`
    - Separate from WebSocket port
    - Prevents false negatives from WebSocket protocol
    
    **Auto-Scaling** (Can be added):
    ```typescript
    const scaling = service.autoScaleTaskCount({
      minCapacity: 1,
      maxCapacity: 10
    });
    
    scaling.scaleOnCpuUtilization('CpuScaling', {
      targetUtilizationPercent: 70
    });
    ```
    
    **Monitoring Dashboard**:
    - CPU and memory utilization
    - Active connections
    - Network throughput
    - Health check status
    - Custom application metrics
    
    **Deployment Strategy**:
    - Blue-green deployments via ECS
    - Connection draining during updates
    - Zero-downtime deployments
    - Rollback capability

20. **Explain your approach to IAM roles and permissions.**
    
    **Answer:** I followed the principle of least privilege throughout:
    
    **Service-Specific Roles**:
    
    1. **Lambda Execution Roles**:
    ```typescript
    const retrievalRole = new iam.Role(this, 'RetrievalRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole')
      ],
      inlinePolicies: {
        BedrockAccess: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: [
                'bedrock:InvokeModel',
                'bedrock:Retrieve',
                'bedrock:RetrieveAndGenerate'
              ],
              resources: ['*']  // Bedrock doesn't support resource-level permissions
            })
          ]
        })
      }
    });
    ```
    
    2. **ECS Task Roles**:
    ```typescript
    taskRole.addToPolicy(new iam.PolicyStatement({
      actions: [
        'bedrock:InvokeModelWithBidirectionalStream',
        'bedrock-agent-runtime:Retrieve'
      ],
      resources: ['*']
    }));
    ```
    
    3. **Cognito Authenticated Role**:
    ```typescript
    authenticatedRole.addToPolicy(new iam.PolicyStatement({
      actions: ['lambda:InvokeFunction'],
      resources: [retrievalFunction.functionArn]  // Specific function only
    }));
    
    authenticatedRole.addToPolicy(new iam.PolicyStatement({
      actions: ['s3:GetObject', 's3:PutObject'],
      resources: [`${mediaBucket.bucketArn}/*`]  // Specific bucket only
    }));
    ```
    
    **Trust Relationships**:
    ```typescript
    const role = new iam.Role(this, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com')
    });
    
    // For Cognito federated access
    const federatedRole = new iam.Role(this, 'FederatedRole', {
      assumedBy: new iam.FederatedPrincipal(
        'cognito-identity.amazonaws.com',
        {
          StringEquals: { 'cognito-identity.amazonaws.com:aud': identityPoolId },
          'ForAnyValue:StringLike': { 'cognito-identity.amazonaws.com:amr': 'authenticated' }
        },
        'sts:AssumeRoleWithWebIdentity'
      )
    });
    ```
    
    **Inline vs Managed Policies**:
    - **Managed**: AWS-managed for common patterns (LambdaBasicExecutionRole)
    - **Inline**: Custom policies specific to application needs
    - Inline policies deleted with role (cleaner)
    
    **Cross-Service Permissions**:
    ```typescript
    // Bedrock Knowledge Base needs multiple services
    kbRole.addToPolicy(new iam.PolicyStatement({
      actions: ['aoss:APIAccessAll'],
      resources: [opensearchCollection.attrArn]
    }));
    
    kbRole.addToPolicy(new iam.PolicyStatement({
      actions: ['s3:GetObject', 's3:ListBucket'],
      resources: [bucket.bucketArn, `${bucket.bucketArn}/*`]
    }));
    
    kbRole.addToPolicy(new iam.PolicyStatement({
      actions: ['bedrock:InvokeModel'],
      resources: [`arn:aws:bedrock:${region}::foundation-model/*`]
    }));
    ```
    
    **Permission Boundaries** (Not implemented but considered):
    - Would limit maximum permissions
    - Useful for delegated administration
    - Prevents privilege escalation
    
    **Best Practices Applied**:
    - Specific resources when possible (not `*`)
    - Minimal actions required
    - Separate roles per service
    - Regular permission audits
    - CloudTrail logging of all API calls
    - IAM Access Analyzer for unused permissions

21. **How do you handle secrets and environment variables?**
    
    **Answer:** I use a layered approach for configuration management:
    
    **SSM Parameter Store**:
    ```typescript
    new ssm.StringParameter(this, 'CognitoUserPoolIdParam', {
      parameterName: `/multimedia-rag/${resourceSuffix}/cognito-user-pool-id`,
      stringValue: userPool.userPoolId,
      description: 'Cognito User Pool ID for Lambda@Edge'
    });
    ```
    - Stores Cognito configuration
    - Accessed by Lambda@Edge at runtime
    - Encrypted at rest
    - Version history
    - IAM-controlled access
    
    **Environment Variables**:
    
    Lambda:
    ```typescript
    environment: {
      MODEL_ID: modelId,
      ORGANIZED_BUCKET: organizedBucket.bucketName,
      IS_BEDROCK_DATA_AUTOMATION: 'true'
    }
    ```
    
    ECS:
    ```typescript
    environment: {
      HOST: '0.0.0.0',
      WS_PORT: '8081',
      AWS_DEFAULT_REGION: 'us-east-1',
      DOCUMENTS_KB_ID: knowledgeBaseId,
      USE_RAG: 'true',
      DEBUG: 'false'
    }
    ```
    
    **Secrets Manager** (For sensitive data - not currently used but available):
    ```typescript
    const secret = new secretsmanager.Secret(this, 'ApiKey', {
      secretName: 'third-party-api-key',
      generateSecretString: {
        secretStringTemplate: JSON.stringify({ username: 'admin' }),
        generateStringKey: 'password'
      }
    });
    
    // Grant read access
    secret.grantRead(lambdaFunction);
    ```
    
    **AWS Credentials**:
    - Never hardcoded
    - IAM roles provide temporary credentials
    - Automatic rotation via STS
    - Boto3 handles credential refresh
    
    **Frontend Configuration**:
    ```javascript
    // React environment variables
    process.env.REACT_APP_USER_POOL_ID
    process.env.REACT_APP_AWS_REGION
    process.env.REACT_APP_WEBSOCKET_URL
    ```
    - Injected at build time
    - Not sensitive (public-facing app)
    - Different values per environment
    
    **Credential Refresh Logic**:
    ```python
    def _refresh_credentials(self):
        session = boto3.Session(region_name=self.region)
        self.credentials = session.get_credentials()
        
        os.environ['AWS_ACCESS_KEY_ID'] = self.credentials.access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = self.credentials.secret_key
        
        if self.credentials.token:
            os.environ['AWS_SESSION_TOKEN'] = self.credentials.token
    ```
    
    **Security Best Practices**:
    - No secrets in code or version control
    - Encryption at rest and in transit
    - Least privilege access to secrets
    - Audit logging via CloudTrail
    - Regular rotation of long-term credentials
    - Separate secrets per environment

22. **What monitoring and logging strategies did you implement?**
    
    **Answer:** Comprehensive observability across all layers:
    
    **CloudWatch Logs**:
    
    Lambda:
    ```typescript
    managedPolicies: [
      iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole')
    ]
    ```
    - Automatic log group creation
    - Structured logging with JSON
    - Log retention policies
    - Log Insights for querying
    
    ECS:
    ```typescript
    logging: ecs.LogDrivers.awsLogs({
      streamPrefix: 'speech-to-speech-backend'
    })
    ```
    - Container stdout/stderr captured
    - Separate log streams per task
    - Real-time log tailing
    
    **S3 Access Logs**:
    ```typescript
    serverAccessLogsBucket: logBucket,
    serverAccessLogsPrefix: 'media-bucket-logs/'
    ```
    - All S3 requests logged
    - Includes requester, operation, response code
    - Useful for security audits
    - Lifecycle rules for cost management
    
    **CloudFront Logs**:
    ```typescript
    enableLogging: true,
    logBucket: accessLogBucket,
    logFilePrefix: 'cloudfront-logs/',
    logIncludesCookies: true
    ```
    - Edge location access logs
    - Request/response details
    - Geographic distribution
    - Performance metrics
    
    **Custom Metrics**:
    ```python
    cloudwatch = boto3.client('cloudwatch')
    cloudwatch.put_metric_data(
        Namespace='MultimediaRAG',
        MetricData=[{
            'MetricName': 'ProcessingDuration',
            'Value': duration,
            'Unit': 'Seconds'
        }]
    )
    ```
    
    **CloudWatch Dashboard**:
    ```typescript
    const dashboard = new cloudwatch.Dashboard(this, 'Dashboard', {
      dashboardName: 'SpeechToSpeech-Metrics'
    });
    
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'ECS Service CPU and Memory',
        left: [
          service.metricCpuUtilization(),
          service.metricMemoryUtilization()
        ]
      })
    );
    ```
    
    **Alarms**:
    ```typescript
    const alarm = new cloudwatch.Alarm(this, 'HighErrorRate', {
      metric: lambdaFunction.metricErrors(),
      threshold: 10,
      evaluationPeriods: 2,
      alarmDescription: 'Alert when error rate is high'
    });
    
    alarm.addAlarmAction(new actions.SnsAction(topic));
    ```
    
    **X-Ray Tracing** (Can be enabled):
    ```typescript
    tracing: lambda.Tracing.ACTIVE
    ```
    - End-to-end request tracing
    - Service map visualization
    - Bottleneck identification
    - Error analysis
    
    **Log Aggregation**:
    - CloudWatch Log Insights queries
    - Athena for S3 access log analysis
    - QuickSight for visualization
    - Export to S3 for long-term storage
    
    **Structured Logging Example**:
    ```python
    logger.info(json.dumps({
        'event': 'file_processed',
        'file_name': filename,
        'duration': duration,
        'status': 'success',
        'timestamp': datetime.now().isoformat()
    }))
    ```
    
    **Monitoring Best Practices**:
    - Log levels: DEBUG, INFO, WARNING, ERROR
    - Correlation IDs for request tracking
    - PII redaction in logs
    - Cost-aware log retention
    - Automated alerting on anomalies
