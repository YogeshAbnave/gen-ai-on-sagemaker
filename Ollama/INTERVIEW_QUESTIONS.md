# Interview Questions Based on Project

## Section 1: Infrastructure as Code & Terraform (Questions 1-10)

### 1. What is Infrastructure as Code and why did you choose Terraform for this project?
**Expected Answer**: Infrastructure as Code (IaC) is the practice of managing infrastructure through code rather than manual processes. I chose Terraform because it's cloud-agnostic, has excellent AWS provider support, uses declarative syntax, maintains state management, and allows version control of infrastructure changes.

### 2. Explain the Terraform workflow you implemented in this project.
**Expected Answer**: The workflow consists of: terraform init (initialize providers), terraform plan (preview changes), terraform apply (create/update resources), and terraform destroy (cleanup). In the GitHub Actions pipeline, I automated this with credential configuration, tfvars generation, and output capture.

### 3. How do you handle Terraform state in your deployment?
**Expected Answer**: Currently using local state stored in the terraform directory. For production, I would recommend migrating to remote state using S3 backend with DynamoDB for state locking to prevent concurrent modifications and enable team collaboration.

### 4. What AWS resources does your Terraform configuration provision?
**Expected Answer**: EC2 instances, security groups, VPC (using default VPC), CloudWatch alarms for monitoring, EBS volumes with encryption, IAM roles (implicitly through instance profiles), and network configurations including public IP assignment.

### 5. How did you make your Terraform configuration flexible and reusable?
**Expected Answer**: I used variables for all configurable parameters (instance_type, storage_size, region, model selection, etc.) with sensible defaults. The configuration supports different instance types, regions, and deployment scenarios through variable overrides.

### 6. Explain the security group configuration in your Terraform code.
**Expected Answer**: The security group allows SSH (port 22) from configurable CIDR, HTTP access to Open-WebUI (port 8080) from anywhere, Ollama API (port 11434) for external access, and all outbound traffic. It's attached to the default VPC and properly tagged.

### 7. What is the purpose of the user-data script in your EC2 configuration?
**Expected Answer**: The user-data script runs on first boot to automate the entire deployment: cloning the repository, installing dependencies (git, Ollama, Docker), downloading AI models, deploying Open-WebUI container, and configuring the system. It includes retry logic and comprehensive logging.

### 8. How do you handle instance metadata in your configuration?
**Expected Answer**: I configured IMDSv2 (Instance Metadata Service version 2) which requires session tokens for metadata access, improving security. This is set through the metadata_options block with http_tokens set to "required".

### 9. What monitoring did you implement using Terraform?
**Expected Answer**: I created CloudWatch alarms for high CPU utilization (threshold 80%, 2 evaluation periods of 5 minutes). The alarm monitors EC2 CPU and can trigger notifications. I also enabled detailed monitoring on the instance for better metrics granularity.

### 10. How would you extend this Terraform configuration for production use?
**Expected Answer**: Add Application Load Balancer, Auto Scaling Groups, multi-AZ deployment, remote state backend (S3+DynamoDB), separate VPC with public/private subnets, NAT gateways, Route53 for DNS, ACM for SSL certificates, and enhanced monitoring with custom CloudWatch dashboards.

## Section 2: CI/CD & GitHub Actions (Questions 11-20)

### 11. Describe the GitHub Actions workflow you implemented for deployment.
**Expected Answer**: The workflow triggers on push to main or manual dispatch. It checks out code, configures AWS credentials, sets up Terraform, generates tfvars from secrets/variables, ensures default VPC exists, runs terraform init/plan/apply, captures outputs, creates deployment artifacts, and posts summaries.

### 12. How do you manage sensitive information in your GitHub Actions workflows?
**Expected Answer**: AWS credentials are stored as GitHub Secrets (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY). Configuration values use GitHub Variables. The workflow never logs sensitive data and uses GitHub's secret masking feature automatically.

### 13. What is the purpose of the destroy workflow and what safety measures did you implement?
**Expected Answer**: The destroy workflow removes all AWS resources to prevent costs. Safety measures include: manual trigger only (workflow_dispatch), required confirmation input (user must type "destroy"), conditional execution checking the confirmation, and comprehensive logging of destruction.

### 14. How do you provide deployment feedback to users in your CI/CD pipeline?
**Expected Answer**: Multiple methods: GitHub Actions summary with formatted output, commit comments with deployment details, artifact uploads (deployment-info.txt), workflow status badges in README, and detailed step-by-step logs in the Actions tab.

### 15. Explain how you handle the terraform.tfvars file in your automated pipeline.
**Expected Answer**: The tfvars file is generated dynamically in the workflow using GitHub Secrets and Variables. This prevents committing sensitive data to the repository while allowing customization. The file is created fresh on each run with current values.

### 16. What would happen if two deployments run simultaneously?
**Expected Answer**: Currently, there's a risk of Terraform state conflicts. To prevent this, I would implement: GitHub Actions concurrency controls, remote state with locking (DynamoDB), or use Terraform Cloud/Enterprise for state management and run queuing.

### 17. How do you ensure the default VPC exists before deployment?
**Expected Answer**: The workflow includes a step that checks for default VPC using AWS CLI. If not found, it creates one using `aws ec2 create-default-vpc`. This prevents deployment failures in regions where the default VPC was deleted.

### 18. What information is captured in the deployment-info artifact?
**Expected Answer**: Deployment timestamp, instance ID, public IP, WebUI URL, SSH commands for accessing the instance, commands to check deployment status and view logs. This artifact provides users with all necessary information to access and troubleshoot their deployment.

### 19. How would you implement a staging environment in this CI/CD pipeline?
**Expected Answer**: Create separate workflows for staging and production, use different AWS accounts or regions, implement branch-based deployments (staging branch → staging env, main → production), use Terraform workspaces or separate state files, and add approval gates for production deployments.

### 20. What testing would you add to this CI/CD pipeline?
**Expected Answer**: Terraform validation and formatting checks, security scanning (tfsec, Checkov), cost estimation (Infracost), integration tests after deployment, health checks to verify services are running, and automated rollback on failure.

## Section 3: AWS & Cloud Architecture (Questions 21-30)

### 21. Why did you choose EC2 over other AWS compute services?
**Expected Answer**: EC2 provides full control over the instance, supports GPU acceleration needed for AI workloads, allows custom kernel tuning, provides persistent storage, and offers flexibility in instance types. Alternatives like ECS/EKS would add complexity for this use case.

### 22. Explain your instance type selection strategy.
**Expected Answer**: Started with t3.xlarge (4 vCPU, 16GB RAM) for development. For production with 3000 users, designed for GPU instances: g4dn.2xlarge (NVIDIA T4, 8 vCPU, 32GB RAM) or g5.xlarge (NVIDIA A10G). Selection based on workload requirements, cost, and availability.

### 23. How does your architecture handle high availability?
**Expected Answer**: Current single-instance design lacks HA. Production design includes: Auto Scaling Groups with min 2 instances, multi-AZ deployment, Application Load Balancer for traffic distribution, health checks with automatic replacement, and EFS for shared storage.

### 24. What is your strategy for handling instance failures?
**Expected Answer**: CloudWatch alarms detect issues, Auto Scaling Groups automatically replace unhealthy instances, load balancers stop routing to failed instances, and user-data scripts ensure new instances are configured identically. Data persistence uses EBS volumes or EFS.

### 25. How do you optimize costs in this AWS deployment?
**Expected Answer**: Auto-scaling to match demand, spot instances for non-critical workloads, gp3 volumes with optimized IOPS, instance right-sizing, automatic shutdown during idle periods, resource tagging for cost allocation, and CloudWatch alarms to prevent runaway costs.

### 26. Explain the networking configuration in your deployment.
**Expected Answer**: Uses default VPC with public subnets, instances get public IPs for direct access, security groups control traffic, all outbound traffic allowed for downloads. Production would use custom VPC with public/private subnets, NAT gateways, and network ACLs.

### 27. How do you handle data persistence in your deployment?
**Expected Answer**: EBS root volume (50GB gp3, encrypted) for OS and applications, Docker volumes (open-webui, chroma-data) for application data. Volumes persist across container restarts. For production, would add EBS snapshots and cross-region replication.

### 28. What security measures did you implement at the AWS level?
**Expected Answer**: Encrypted EBS volumes, IMDSv2 for metadata access, security groups with minimal required ports, SSH access restriction via CIDR, IAM roles with least privilege, VPC isolation, and resource tagging for governance.

### 29. How would you implement disaster recovery for this system?
**Expected Answer**: Automated EBS snapshots, cross-region replication, infrastructure code in version control for quick rebuild, documented recovery procedures, regular backup testing, RTO/RPO definitions, and multi-region deployment for critical workloads.

### 30. Explain how you would monitor costs for this deployment.
**Expected Answer**: AWS Cost Explorer for analysis, CloudWatch billing alarms, resource tagging (project, environment), AWS Budgets for spending limits, Cost Allocation Tags, and regular cost optimization reviews using AWS Trusted Advisor and Compute Optimizer.

## Section 4: Docker & Containerization (Questions 31-40)

### 31. Why did you containerize Open-WebUI using Docker?
**Expected Answer**: Docker provides isolation, consistent environments across deployments, easy updates and rollbacks, dependency management, resource limits, and simplified deployment. Open-WebUI's official distribution is containerized, making it the natural choice.

### 32. Explain the Docker networking configuration in your deployment.
**Expected Answer**: Using `--network host` mode so the container shares the host's network namespace. This simplifies communication with Ollama (localhost:11434) and exposes port 8080 directly. Alternative would be bridge networking with explicit port mappings.

### 33. What are Docker volumes and how do you use them?
**Expected Answer**: Docker volumes provide persistent storage independent of container lifecycle. I use `open-webui` volume for application data and `chroma-data` volume for vector database storage. Data persists across container restarts and updates.

### 34. How do you handle Docker container updates in your deployment?
**Expected Answer**: Pull latest image (`ghcr.io/open-webui/open-webui:main`), stop existing container, remove it, start new container with same volumes and configuration. Volumes ensure data persistence. Would implement blue-green deployment for zero-downtime updates.

### 35. What is the purpose of `--restart always` in your Docker run command?
**Expected Answer**: Ensures the container automatically restarts if it crashes or if the host reboots. This provides resilience without manual intervention. Docker daemon will continuously attempt to restart the container until explicitly stopped.

### 36. How do you pass configuration to the Open-WebUI container?
**Expected Answer**: Using environment variables (-e flags): OLLAMA_BASE_URL for Ollama connection, RAG configuration (embedding model, top-k, chunk size), and feature flags. This follows 12-factor app principles for configuration management.

### 37. What is the purpose of `--add-host=host.docker.internal:host-gateway`?
**Expected Answer**: Creates a DNS entry that resolves to the host machine's IP from within the container. This allows the container to access services running on the host (like Ollama) using a consistent hostname regardless of the actual host IP.

### 38. How would you implement container health checks?
**Expected Answer**: Add HEALTHCHECK instruction in Dockerfile or use `--health-cmd` flag. Check HTTP endpoint availability (curl localhost:8080), verify Ollama connectivity, check disk space. Docker can automatically restart unhealthy containers.

### 39. Explain your Docker image selection strategy.
**Expected Answer**: Using official Open-WebUI image from GitHub Container Registry (ghcr.io), using `main` tag for latest stable version. For production, would pin specific version tags for reproducibility and implement image scanning for vulnerabilities.

### 40. How do you troubleshoot Docker container issues?
**Expected Answer**: Check container status (`docker ps`), view logs (`docker logs -f open-webui`), inspect container (`docker inspect`), execute commands inside container (`docker exec -it`), check resource usage (`docker stats`), and verify volume mounts.

## Section 5: AI/ML & Ollama (Questions 41-50)

### 41. What is Ollama and why did you choose it for this project?
**Expected Answer**: Ollama is a lightweight AI model runtime that makes it easy to run large language models locally. I chose it for its simplicity, GPU support, model management capabilities, REST API, and active community. It handles model loading, inference, and resource management.

### 42. How does Ollama utilize GPU acceleration?
**Expected Answer**: Ollama automatically detects NVIDIA GPUs when CUDA drivers are installed. It loads models into GPU memory for faster inference. The user-data script installs NVIDIA drivers, and Ollama's snap package includes CUDA support for seamless GPU utilization.

### 43. Explain the AI models you configured in this deployment.
**Expected Answer**: Configured multiple models: DeepSeek-R1 (8b, 14b, 32b variants) for reasoning, Llama 3.2 (3b, 8b) for general use, and Qwen 2.5 (7b) for multilingual support. Model selection via DEFAULT_MODEL variable. Each has different size/performance tradeoffs.

### 44. How do you handle model downloads in your deployment?
**Expected Answer**: User-data script automatically downloads the selected model using `ollama run` command. Models are cached locally (~5-20GB depending on model). First run triggers download, subsequent runs use cached model. Implements retry logic for reliability.

### 45. What is the difference between model parameters (8b, 14b, 32b)?
**Expected Answer**: Numbers indicate billions of parameters. More parameters generally mean better quality but require more memory and compute. 8b models (~5GB) run on smaller instances, 32b models (~20GB) need more resources. Tradeoff between quality and performance.

### 46. How does Open-WebUI communicate with Ollama?
**Expected Answer**: Open-WebUI connects to Ollama via REST API on localhost:11434. Configuration passed via OLLAMA_BASE_URL environment variable. API handles model listing, chat completions, embeddings generation, and streaming responses.

### 47. What is inference latency and how do you optimize it?
**Expected Answer**: Time from query submission to response generation. Optimizations: GPU acceleration (2-10x faster), model selection (smaller models faster), batch processing, model caching in memory, kernel tuning (memory management, network optimization), and instance type selection.

### 48. How do you handle concurrent users in your AI deployment?
**Expected Answer**: Ollama queues requests and processes them based on available resources. GPU instances handle more concurrent requests. For 3000 users, designed auto-scaling with multiple instances behind load balancer, each handling subset of traffic.

### 49. What monitoring would you implement for AI model performance?
**Expected Answer**: Track inference latency (p50, p95, p99), requests per second, GPU utilization, memory usage, model loading time, error rates, queue depth, and user satisfaction metrics. Use CloudWatch custom metrics and dashboards.

### 50. How would you implement A/B testing for different AI models?
**Expected Answer**: Deploy multiple instances with different models, use load balancer weighted routing, track performance metrics per model, collect user feedback, analyze quality vs. cost tradeoffs, and gradually shift traffic to better-performing models.

## Section 6: RAG (Retrieval-Augmented Generation) (Questions 51-60)

### 51. What is RAG and why is it important?
**Expected Answer**: Retrieval-Augmented Generation combines document retrieval with AI generation. It allows AI to reference specific documents when answering, improving accuracy, reducing hallucinations, providing source attribution, and enabling domain-specific knowledge without retraining models.

### 52. Explain the RAG architecture you implemented.
**Expected Answer**: Users upload documents → parsed and chunked → embedded using sentence-transformers → stored in ChromaDB vector database → queries embedded → similarity search retrieves relevant chunks → chunks passed as context to Ollama → AI generates response with sources.

### 53. What is a vector database and why did you choose ChromaDB?
**Expected Answer**: Vector databases store and query high-dimensional embeddings efficiently. ChromaDB chosen for: embedded mode (no separate server), Python-native, good performance, persistent storage, simple API, and excellent Open-WebUI integration.

### 54. How does document chunking work in your RAG implementation?
**Expected Answer**: Documents split into 512-token chunks with 50-token overlap. Overlap ensures context continuity across chunks. Recursive character splitter preserves sentence boundaries. Chunk size balances context richness vs. retrieval precision.

### 55. What is an embedding model and which one did you use?
**Expected Answer**: Embedding models convert text to numerical vectors capturing semantic meaning. Used sentence-transformers/all-MiniLM-L6-v2: 384 dimensions, ~80MB size, fast inference (~50ms), good quality-performance balance, runs locally without external APIs.

### 56. Explain the document upload and indexing process.
**Expected Answer**: User uploads file → Open-WebUI validates format → extracts text (PyPDF2 for PDF, python-docx for DOCX) → chunks text → generates embeddings → stores in ChromaDB with metadata (source, page, position) → confirms to user. Async for large documents.

### 57. How does semantic search work in your RAG system?
**Expected Answer**: Query embedded using same model → ChromaDB performs cosine similarity search → returns top-k most similar chunks (default 5) → filters by similarity threshold (0.7) → chunks ranked by relevance → passed to AI as context.

### 58. What file formats does your RAG system support and how?
**Expected Answer**: PDF (PyPDF2/pdfplumber), TXT (direct), Markdown (markdown parser), DOCX (python-docx). Each parser extracts text while preserving structure. Unsupported formats rejected with clear error. Extensible via plugin architecture.

### 59. How do you handle RAG system resource constraints?
**Expected Answer**: Vector database memory limit (2GB), batch embedding processing (32 chunks), lazy model loading, disk-based persistence, query result caching, LRU eviction for caches, and async processing for large documents.

### 60. What correctness properties did you define for the RAG system?
**Expected Answer**: Document upload completeness, embedding consistency, retrieval relevance ordering, source attribution accuracy, deletion completeness, configuration bounds enforcement, resource limits, format support validation, chunk overlap preservation, and retrieval idempotence.

## Section 7: System Administration & Linux (Questions 61-70)

### 61. What kernel tuning did you implement and why?
**Expected Answer**: Optimized for AI workloads: reduced swappiness (10) for memory preference, increased dirty ratios for write performance, raised max_map_count for memory mapping, enabled overcommit for large allocations, tuned network buffers, increased file limits, and configured shared memory for large models.

### 62. How do you handle logging in your deployment?
**Expected Answer**: User-data script logs to /var/log/user-data.log with structured format (timestamp, level, component, message). Docker logs accessible via `docker logs`. CloudWatch Logs for centralized logging. Log rotation configured to prevent disk filling.

### 63. What is the purpose of the deployment-status.txt file?
**Expected Answer**: JSON file tracking deployment progress: overall status, timestamp, component statuses (git, ollama, docker, model, webui), error messages, WebUI URL, and log file location. Allows users to quickly check deployment state without parsing logs.

### 64. How do you handle service failures and restarts?
**Expected Answer**: Ollama installed via snap with automatic restart, Docker containers with `--restart always`, systemd for service management, health checks for detection, CloudWatch alarms for alerting, and Auto Scaling for instance replacement.

### 65. Explain your approach to system security hardening.
**Expected Answer**: Minimal installed packages, automatic security updates, encrypted volumes, IMDSv2, restricted SSH access, non-root container execution where possible, security groups as firewall, disabled unnecessary services, and regular security scanning.

### 66. How do you manage system updates and patches?
**Expected Answer**: Ubuntu unattended-upgrades for security patches, snap auto-refresh for Ollama, Docker image updates via container replacement, Terraform for infrastructure updates, and scheduled maintenance windows for major updates.

### 67. What is your backup and recovery strategy?
**Expected Answer**: EBS snapshots for volumes, Docker volume backups, infrastructure code in git for rebuild, documented recovery procedures, tested restore process, and automated snapshot scheduling via AWS Backup or Lambda.

### 68. How do you troubleshoot performance issues?
**Expected Answer**: Check CPU/memory/disk with `top`/`htop`, GPU utilization with `nvidia-smi`, network with `netstat`/`ss`, Docker stats, application logs, CloudWatch metrics, and profiling tools. Systematic approach: identify bottleneck → analyze → optimize → verify.

### 69. What is your approach to capacity planning?
**Expected Answer**: Monitor current usage patterns, project growth, load testing to find limits, calculate resources per user, plan for peak loads with headroom, implement auto-scaling for elasticity, and regular capacity reviews.

### 70. How do you implement zero-downtime deployments?
**Expected Answer**: Blue-green deployment with two environments, load balancer switches traffic, health checks verify new version, gradual traffic shift, quick rollback capability, database migration strategies, and session persistence handling.

## Section 8: Security & Compliance (Questions 71-80)

### 71. What security best practices did you implement in this project?
**Expected Answer**: Encrypted storage, IMDSv2, least privilege IAM, security groups, SSH key authentication, secrets management via GitHub Secrets, no hardcoded credentials, HTTPS for production, regular updates, and security scanning.

### 72. How do you manage SSH keys in your deployment?
**Expected Answer**: SSH key pair created in AWS, private key stored securely (not in repo), key name configurable via variable, restricted SSH access via security group CIDR, key rotation policy, and alternative access via AWS Systems Manager Session Manager.

### 73. What is IMDSv2 and why is it important?
**Expected Answer**: Instance Metadata Service version 2 requires session tokens for metadata access, preventing SSRF attacks. Configured via metadata_options in Terraform. More secure than IMDSv1 which allowed unauthenticated access.

### 74. How do you handle secrets management?
**Expected Answer**: AWS credentials in GitHub Secrets, no secrets in code or logs, environment variables for configuration, AWS Secrets Manager for production, secret rotation policies, and audit logging of secret access.

### 75. What compliance considerations would you address for production?
**Expected Answer**: Data encryption at rest and in transit, access logging and auditing, data retention policies, GDPR compliance for user data, SOC 2 requirements, regular security assessments, and incident response procedures.

### 76. How do you implement network security?
**Expected Answer**: Security groups as stateful firewall, network ACLs for subnet-level control, private subnets for backend, VPC flow logs for monitoring, AWS WAF for application protection, and DDoS protection via AWS Shield.

### 77. What is your approach to vulnerability management?
**Expected Answer**: Regular security scanning (Trivy for containers, tfsec for Terraform), automated patching, vulnerability tracking, CVE monitoring, dependency updates, security advisories subscription, and penetration testing.

### 78. How do you implement access control?
**Expected Answer**: IAM roles with least privilege, MFA for AWS console, SSH key-based authentication, Open-WebUI user authentication, role-based access control (RBAC), audit logging, and regular access reviews.

### 79. What logging and auditing do you implement for security?
**Expected Answer**: CloudTrail for API calls, VPC Flow Logs for network traffic, application logs, access logs, security group changes, failed authentication attempts, and centralized log analysis with retention policies.

### 80. How would you implement data privacy in the RAG system?
**Expected Answer**: User data isolation, encrypted storage, access controls, data deletion capabilities, privacy policy compliance, no external API calls for embeddings (local processing), and audit trails for data access.

## Section 9: DevOps & Best Practices (Questions 81-90)

### 81. What is GitOps and how did you implement it?
**Expected Answer**: GitOps uses Git as single source of truth for infrastructure and applications. Every push triggers automated deployment, infrastructure defined in code, version controlled, peer review via PRs, audit trail in git history, and easy rollback.

### 82. How do you implement code review in your workflow?
**Expected Answer**: Pull requests for changes, required reviews before merge, automated checks (linting, validation), branch protection rules, CI runs on PRs, and documented review guidelines.

### 83. What is your branching strategy?
**Expected Answer**: Main branch for production, feature branches for development, PR-based merges, automated deployment from main, tags for releases, and hotfix branches for urgent fixes.

### 84. How do you handle environment-specific configurations?
**Expected Answer**: Terraform variables for differences, separate tfvars files per environment, GitHub environments for secrets, naming conventions (project-env), and workspace isolation.

### 85. What documentation did you create for this project?
**Expected Answer**: README with quick start, architecture diagrams, troubleshooting guides, API documentation, runbooks for operations, inline code comments, and specification documents (requirements, design, tasks).

### 86. How do you ensure deployment reproducibility?
**Expected Answer**: Infrastructure as code, version pinning (Terraform, Docker images), documented procedures, automated deployment, no manual steps, and tested recovery procedures.

### 87. What is your approach to testing infrastructure code?
**Expected Answer**: Terraform validation, plan review, test deployments in dev environment, automated testing (Terratest), security scanning, cost estimation, and post-deployment verification.

### 88. How do you implement observability?
**Expected Answer**: Metrics (CloudWatch), logs (centralized logging), traces (X-Ray for distributed tracing), dashboards for visualization, alerts for anomalies, and SLI/SLO definitions.

### 89. What is your incident response process?
**Expected Answer**: Detection via monitoring, alert routing, incident classification, documented runbooks, communication plan, root cause analysis, post-mortems, and continuous improvement.

### 90. How do you measure deployment success?
**Expected Answer**: Deployment time, success rate, rollback frequency, MTTR (mean time to recovery), user impact, cost efficiency, and automated verification tests.

## Section 10: Advanced Topics & Future Improvements (Questions 91-100)

### 91. How would you implement multi-region deployment?
**Expected Answer**: Terraform modules per region, Route53 for DNS routing, cross-region replication for data, global load balancing, latency-based routing, disaster recovery procedures, and cost optimization across regions.

### 92. What caching strategies would you implement?
**Expected Answer**: CloudFront for static content, Redis for application caching, query result caching in RAG, model output caching, DNS caching, and CDN for global distribution.

### 93. How would you implement rate limiting?
**Expected Answer**: API Gateway for request throttling, application-level rate limiting, per-user quotas, DDoS protection, graceful degradation under load, and queue-based request handling.

### 94. What database would you add for user management?
**Expected Answer**: RDS PostgreSQL for relational data, DynamoDB for NoSQL needs, Aurora for high availability, automated backups, read replicas for scaling, and connection pooling.

### 95. How would you implement model versioning?
**Expected Answer**: Model registry, version tagging, A/B testing framework, gradual rollout, performance comparison, rollback capability, and model metadata tracking.

### 96. What would you do to improve cost efficiency?
**Expected Answer**: Reserved instances for baseline, spot instances for batch workloads, auto-scaling for elasticity, right-sizing instances, S3 lifecycle policies, CloudWatch cost anomaly detection, and regular cost reviews.

### 97. How would you implement blue-green deployment?
**Expected Answer**: Two identical environments, load balancer switches traffic, health checks verify new version, database migration strategy, quick rollback, and automated testing before switch.

### 98. What observability improvements would you add?
**Expected Answer**: Distributed tracing (X-Ray), custom metrics, user experience monitoring, error tracking (Sentry), performance profiling, log aggregation (ELK stack), and real-time dashboards.

### 99. How would you implement feature flags?
**Expected Answer**: Feature toggle service (LaunchDarkly, AWS AppConfig), gradual rollout, A/B testing, user segmentation, kill switches for problematic features, and analytics integration.

### 100. What machine learning operations (MLOps) practices would you implement?
**Expected Answer**: Model versioning, experiment tracking, automated retraining, model monitoring, performance degradation detection, data drift detection, model registry, and CI/CD for models.

---

## Bonus Questions (101-105)

### 101. Explain the specification-driven development approach you used.
**Expected Answer**: Created formal requirements with EARS patterns, designed with correctness properties, planned implementation tasks, property-based testing for verification, and iterative refinement with user feedback.

### 102. What are correctness properties and why are they important?
**Expected Answer**: Universal statements about system behavior that should hold for all inputs. They bridge specifications and verification, enable property-based testing, catch edge cases, and provide formal guarantees about correctness.

### 103. How did you design for scalability from 100 to 3000 users?
**Expected Answer**: Auto Scaling Groups, load balancing, GPU instances for performance, horizontal scaling, stateless application design, distributed caching, database read replicas, and performance testing at scale.

### 104. What trade-offs did you make in this project?
**Expected Answer**: Simplicity vs. features (started simple), cost vs. performance (right-sized instances), speed vs. reliability (automated deployment with safety checks), and flexibility vs. complexity (configurable but not overwhelming).

### 105. How would you mentor someone learning this technology stack?
**Expected Answer**: Start with fundamentals (Linux, networking), hands-on projects, incremental complexity, code reviews, documentation reading, community engagement, troubleshooting practice, and continuous learning mindset.
