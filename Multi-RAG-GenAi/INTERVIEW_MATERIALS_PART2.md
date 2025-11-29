# Interview Materials Part 2: Detailed Answers (Continued)

## Backend Development (10 questions)

23. **Walk me through your WebSocket server implementation.**
    
    **Answer:** The WebSocket server is built with Python's `websockets` library and asyncio:
    
    **Server Setup**:
    ```python
    async def main(host, port, health_port):
        server_config = {
            "ping_interval": 20,  # Send ping every 20 seconds
            "ping_timeout": 30,    # Wait 30 seconds for pong
            "close_timeout": 10    # Wait 10 seconds for close handshake
        }
        
        async with websockets.serve(authenticated_handler, host, port, **server_config):
            logger.info(f"WebSocket server started at {host}:{port}")
            await asyncio.Future()  # Run forever
    ```
    
    **Connection Handler**:
    ```python
    async def websocket_handler(websocket, path=None):
        stream_manager = None
        connection_id = f"conn-{int(time.time())}"
        
        # Extract and validate JWT token from query string
        if '?' in path:
            query_string = path.split('?', 1)[1]
            params = urllib.parse.parse_qs(query_string)
            token = params.get('token', [None])[0]
            
            valid, user_id, username = await validate_token(token, client_ip)
            if not valid:
                await websocket.close(1008, "Invalid authentication token")
                return
        
        # Initialize stream manager
        stream_manager = S2sSessionManager(model_id='amazon.nova-sonic-v1:0')
        await stream_manager.initialize_stream()
        
        # Start forwarding responses
        forward_task = asyncio.create_task(
            forward_responses(websocket, stream_manager, connection_id)
        )
        
        # Process incoming messages
        async for message in websocket:
            data = json.loads(message)
            if 'event' in data:
                event_type = list(data['event'].keys())[0]
                
                if event_type == 'audioInput':
                    stream_manager.add_audio_chunk(...)
                else:
                    await stream_manager.send_raw_event(data)
    ```
    
    **Authentication**:
    - JWT token in query parameter (WebSocket doesn't support headers well)
    - Validates against Cognito JWKS
    - RSA signature verification
    - Expiration and issuer checks
    - Connection rejected if invalid
    
    **Message Routing**:
    - Audio events → audio queue for processing
    - Control events → sent directly to Bedrock
    - Tool use events → processed and results returned
    
    **Graceful Shutdown**:
    ```python
    finally:
        if stream_manager:
            await stream_manager.close()
        if forward_task:
            forward_task.cancel()
        if websocket:
            websocket.close()
    ```
    
    **Health Check Server**:
    - Separate HTTP server on port 8082
    - Returns JSON: `{"status": "healthy"}`
    - Used by NLB health checks
    - Runs in separate thread
    
    **Error Handling**:
    - Connection errors logged with context
    - Automatic cleanup on disconnect
    - Credential refresh on auth errors
    - Graceful degradation

24. **How does the S2sSessionManager class work?**
    
    **Answer:** S2sSessionManager manages the bidirectional stream with Bedrock:
    
    **Initialization**:
    ```python
    class S2sSessionManager:
        def __init__(self, model_id='amazon.nova-sonic-v1:0', region='us-east-1'):
            self.model_id = model_id
            self.region = region
            self.audio_input_queue = asyncio.Queue()
            self.output_queue = asyncio.Queue()
            self.bedrock_client = None
            self.stream = None
            self.is_active = False
            
            # Credential management
            self.credentials = None
            self.credentials_expiration = None
            self.credential_refresh_task = None
    ```
    
    **Stream Initialization**:
    ```python
    async def initialize_stream(self):
        # Refresh credentials if needed
        if not self.bedrock_client or self._should_refresh_credentials():
            self._initialize_client()
        
        # Start periodic credential refresh
        self.credential_refresh_task = asyncio.create_task(
            self._periodic_credential_refresh()
        )
        
        # Initialize bidirectional stream
        self.stream = await self.bedrock_client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.is_active = True
        
        # Start response processing
        self.response_task = asyncio.create_task(self._process_responses())
        
        # Start audio input processing
        asyncio.create_task(self._process_audio_input())
    ```
    
    **Audio Processing**:
    ```python
    async def _process_audio_input(self):
        while self.is_active:
            # Get audio from queue
            data = await self.audio_input_queue.get()
            
            # Create audio input event
            audio_event = {
                "event": {
                    "audioInput": {
                        "promptName": data['prompt_name'],
                        "contentName": data['content_name'],
                        "content": data['audio_bytes']
                    }
                }
            }
            
            # Send to Bedrock
            await self.send_raw_event(audio_event)
    ```
    
    **Response Processing**:
    ```python
    async def _process_responses(self):
        while self.is_active:
            output = await self.stream.await_output()
            result = await output[1].receive()
            
            if result.value and result.value.bytes_:
                response_data = result.value.bytes_.decode('utf-8')
                json_data = json.loads(response_data)
                json_data["timestamp"] = int(time.time() * 1000)
                
                # Handle tool use
                if event_name == 'toolUse':
                    self.toolName = json_data['event']['toolUse']['toolName']
                    self.toolUseId = json_data['event']['toolUse']['toolUseId']
                
                # Process tool use when content ends
                elif event_name == 'contentEnd' and type == 'TOOL':
                    toolResult = await self.processToolUse(self.toolName, self.toolUseContent)
                    # Send tool result events...
                
                # Forward to client
                await self.output_queue.put(json_data)
    ```
    
    **Credential Management**:
    ```python
    def _refresh_credentials(self):
        session = boto3.Session(region_name=self.region)
        self.credentials = session.get_credentials()
        
        # Update environment variables for Smithy
        os.environ['AWS_ACCESS_KEY_ID'] = self.credentials.access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = self.credentials.secret_key
        if self.credentials.token:
            os.environ['AWS_SESSION_TOKEN'] = self.credentials.token
        
        # Track expiration
        if hasattr(self.credentials, 'expiry_time'):
            self.credentials_expiration = self.credentials.expiry_time
    
    async def _periodic_credential_refresh(self):
        while self.is_active:
            await asyncio.sleep(3600)  # Check every hour
            if self._should_refresh_credentials():
                self._refresh_credentials()
    ```
    
    **Tool Use Processing**:
    ```python
    async def processToolUse(self, toolName, toolUseContent):
        if toolName == "getKbTool":
            query = self._extract_query(toolUseContent)
            
            if os.environ.get('USE_RAG') == 'true':
                results = kb.retrieve_and_generation(query)
            else:
                results = kb.retrieve_kb(query)
            
            return {"result": "\n\n".join(results)}
    ```
    
    **Cleanup**:
    ```python
    async def close(self):
        self.is_active = False
        
        if self.credential_refresh_task:
            self.credential_refresh_task.cancel()
        
        if self.stream:
            await self.stream.input_stream.close()
        
        if self.response_task:
            self.response_task.cancel()
    ```

25. **Explain your audio processing pipeline in the speech-to-speech feature.**
    
    **Answer:** The audio pipeline handles real-time capture, encoding, and streaming:
    
    **Frontend Capture**:
    ```javascript
    async startMicrophone() {
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });
        
        // Create audio context
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({
            latencyHint: 'interactive'
        });
        
        const source = audioContext.createMediaStreamSource(stream);
        const processor = audioContext.createScriptProcessor(512, 1, 1);
        
        source.connect(processor);
        processor.connect(audioContext.destination);
    }
    ```
    
    **Resampling to 16kHz**:
    ```javascript
    processor.onaudioprocess = async (e) => {
        const inputBuffer = e.inputBuffer;
        const targetSampleRate = 16000;
        
        // Create offline context for resampling
        const offlineContext = new OfflineAudioContext({
            numberOfChannels: 1,
            length: Math.ceil(inputBuffer.duration * targetSampleRate),
            sampleRate: targetSampleRate
        });
        
        // Copy and resample
        const offlineSource = offlineContext.createBufferSource();
        const monoBuffer = offlineContext.createBuffer(1, inputBuffer.length, inputBuffer.sampleRate);
        monoBuffer.copyToChannel(inputBuffer.getChannelData(0), 0);
        
        offlineSource.buffer = monoBuffer;
        offlineSource.connect(offlineContext.destination);
        offlineSource.start(0);
        
        const renderedBuffer = await offlineContext.startRendering();
        const resampled = renderedBuffer.getChannelData(0);
    }
    ```
    
    **PCM Encoding**:
    ```javascript
    // Convert Float32 to Int16 PCM
    const buffer = new ArrayBuffer(resampled.length * 2);
    const pcmData = new DataView(buffer);
    
    for (let i = 0; i < resampled.length; i++) {
        const s = Math.max(-1, Math.min(1, resampled[i]));
        pcmData.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    ```
    
    **Base64 Encoding**:
    ```javascript
    // Convert to binary string
    let binary = '';
    for (let i = 0; i < pcmData.byteLength; i++) {
        binary += String.fromCharCode(pcmData.getUint8(i));
    }
    
    // Base64 encode
    const base64Audio = btoa(binary);
    ```
    
    **WebSocket Transmission**:
    ```javascript
    const event = S2sEvent.audioInput(
        this.state.promptName,
        this.state.audioContentName,
        base64Audio
    );
    this.sendEvent(event);
    ```
    
    **Backend Processing**:
    ```python
    # Decode base64
    audio_bytes = base64.b64decode(audio_data)
    
    # Queue for processing
    self.audio_input_queue.put_nowait({
        'prompt_name': prompt_name,
        'content_name': content_name,
        'audio_bytes': audio_data
    })
    
    # Send to Bedrock
    await self.send_raw_event(audio_event)
    ```
    
    **Response Playback**:
    ```javascript
    handleIncomingMessage(message) {
        if (eventType === "audioOutput") {
            // Accumulate audio chunks
            audioResponse[contentId] += message.event.audioOutput.content;
        }
        else if (eventType === "contentEnd" && contentType === "AUDIO") {
            // Convert base64 LPCM to playable audio
            const audioUrl = base64LPCM(this.state.audioResponse[contentId]);
            
            // Queue for sequential playback
            this.audioEnqueue(audioUrl);
        }
    }
    
    playNext() {
        if (this.audioQueue.length > 0) {
            const audioUrl = this.audioQueue.shift();
            this.audioPlayerRef.current.src = audioUrl;
            this.audioPlayerRef.current.play();
            
            this.audioPlayerRef.current.onended = () => {
                this.playNext();  // Play next in queue
            };
        }
    }
    ```
    
    **Interruption Handling**:
    ```javascript
    // Detect interruption signal
    if (role === "ASSISTANT" && content.startsWith("{")) {
        const evt = JSON.parse(content);
        if (evt.interrupted === true) {
            this.cancelAudio();  // Stop current playback
        }
    }
    
    cancelAudio() {
        if (this.audioPlayerRef.current) {
            this.audioPlayerRef.current.pause();
            this.audioPlayerRef.current.currentTime = 0;
        }
        this.audioQueue = [];  // Clear queue
    }
    ```
    
    **Performance Optimizations**:
    - Small buffer size (512 samples) for low latency
    - Efficient resampling using OfflineAudioContext
    - Queue-based processing prevents blocking
    - Sequential playback prevents audio overlap
    - Interruption support for natural conversation

(Continue with remaining questions in similar detail...)

## Frontend Development, Infrastructure, Security, and Problem-Solving sections would follow with equally detailed answers covering:

- React component architecture
- State management with hooks and context
- AWS SDK integration
- File upload handling
- CDK testing and deployment
- Security best practices
- Debugging strategies
- Performance optimization
- Code quality practices

Each answer provides:
- Code examples
- Architecture decisions
- Trade-offs considered
- Best practices applied
- Real-world scenarios
- Lessons learned
