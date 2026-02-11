import Foundation

@available(macOS 10.15, *)
internal final actor LLMManager {
    static let shared = LLMManager()
    
    private var modelPtr: OpaquePointer?
    private var ctxPtr: OpaquePointer?
    private var samplerPtr: OpaquePointer?
    private var isInitialized = false
    private let config: LLMConfig
    private var customSystemPrompt: String? = nil
    
    private init() {
        self.config = LLMConfig()
    }
    
    func setCustomSystemPrompt(_ prompt: String?) {
        customSystemPrompt = prompt
    }
    
    func initialize(modelPath: String, threads: Int32, contextSize: Int32, customSystemPrompt: String? = nil) throws {
        guard !isInitialized else { return }
        
        llama_backend_init()
        
        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = 0
        modelParams.use_mmap = true
        
        modelPtr = try loadModel(path: modelPath, params: &modelParams)
        guard modelPtr != nil else {
            llama_backend_free()
            throw LLMError.modelLoadFailed
        }
        
        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = UInt32(contextSize)
        ctxParams.n_threads = UInt32(threads)
        
        ctxPtr = try createContext(threads: threads, contextSize: contextSize, params: &ctxParams)
        guard ctxPtr != nil else {
            llama_model_free(modelPtr)
            modelPtr = nil
            llama_backend_free()
            throw LLMError.contextCreationFailed
        }
        
        var samplerParams = llama_sampler_chain_default_params()
        samplerParams.temperature = 0.7
        samplerParams.top_k = 40
        samplerParams.top_p = 0.95
        
        samplerPtr = llama_sampler_chain_init(ctxPtr, &samplerParams)
        let seqId: Int32 = 0
        llama_set_sampler(ctxPtr, seqId, samplerPtr)
        
        isInitialized = true
    }
    
    func correctText(_ text: String) async throws -> String {
        guard isInitialized, let ctx = ctxPtr else {
            print("[DEBUG] LLM not initialized, throwing error")
            fflush(stdout)
            throw LLMError.notInitialized
        }
        
        let systemPrompt = customSystemPrompt ?? LLMConstants.systemPrompt
        let userPrompt = LLMConstants.correctionPrompt + text
        let fullPrompt = "System:\n\(systemPrompt)\n\nUser:\n\(userPrompt)\n\nAssistant:"
        
        let usingCustom = customSystemPrompt != nil
        print("[DEBUG] --- プロンプト構築開始 ---")
        print("[DEBUG] Using custom system prompt: \(usingCustom)")
        print("[DEBUG] System prompt length: \(systemPrompt.count) chars")
        print("[DEBUG] User prompt length: \(userPrompt.count) chars")
        print("[DEBUG] Full prompt length: \(fullPrompt.count) chars")
        print("[DEBUG] Full prompt (first 300 chars): \(fullPrompt.prefix(300))...")
        print("[DEBUG] --- プロンプト構築完了 ---")
        fflush(stdout)
        
        print("[DEBUG] Starting LLM generation...")
        fflush(stdout)
        
        let result = try await generateText(prompt: fullPrompt, ctx: ctx)
        
        print("[DEBUG] LLM generation completed: \"\(result)\"")
        fflush(stdout)
        
        return result
    }
    
    func shutdown() {
        guard isInitialized else { return }
        
        if let sampler = samplerPtr {
            llama_sampler_free(sampler)
            samplerPtr = nil
        }
        
        if let ctx = ctxPtr {
            llama_free(ctx)
            ctxPtr = nil
        }
        
        if let model = modelPtr {
            llama_model_free(model)
            modelPtr = nil
        }
        
        llama_backend_free()
        isInitialized = false
    }
    
    private func loadModel(path: String, params: UnsafePointer<llama_model_params>) throws -> OpaquePointer? {
        return path.withCString { cPath in
            llama_model_load_from_file(cPath, params)
        }
    }
    
    private func createContext(threads: Int32, contextSize: Int32, params: UnsafePointer<llama_context_params>) throws -> OpaquePointer? {
        guard let model = modelPtr else {
            throw LLMError.modelLoadFailed
        }
        return llama_init_from_model(model, params)
    }
    
    private func generateText(prompt: String, ctx: OpaquePointer) async throws -> String {
        var result = ""
        
        try prompt.withCString { cPrompt in
            let promptLen = Int32(strlen(cPrompt))
            var tokens: [llama_token] = Array(repeating: 0, count: 4096)
            
            let tokenCount = llama_tokenize(ctx, cPrompt, promptLen, &tokens, 4096, true, false)
            
            let eos = llama_token_eos(ctx)
            
            print("[DEBUG] Tokenizing prompt...")
            print("[DEBUG] Prompt length: \(promptLen) chars, \(tokenCount) tokens")
            fflush(stdout)
            
            var batch = llama_batch_init(Int32(min(tokenCount, 512)), 0, 1)
            defer { llama_batch_free(batch) }
            
            if let batchToken = batch.token, let batchPos = batch.pos, let batchNS = batch.n_seq_id, let batchSeq = batch.seq_id, let batchLogits = batch.logits {
                for i in 0..<Int(tokenCount) {
                    batch.n_tokens = Int32(i + 1)
                    batchToken.advanced(by: i).pointee = tokens[i]
                    batchPos.advanced(by: i).pointee = Int32(i)
                    batchNS.advanced(by: i).pointee = 1
                    batchSeq.advanced(by: i).pointee?.advanced(by: 0).pointee = 0
                    batchLogits.advanced(by: i).pointee = (i == Int(tokenCount) - 1) ? 1 : 0
                    
                    let ret = llama_decode(ctx, &batch)
                    if ret != 0 {
                        throw LLMError.generationFailed
                    }
                }
                
                var generated = 0
                let maxGenerated = 2048
                
                print("[DEBUG] Starting generation, max tokens: \(maxGenerated)...")
                fflush(stdout)
                
                while generated < maxGenerated {
                    var token = llama_get_sampled_token_ith(ctx, 0)
                    
                    if token == eos {
                        print("[DEBUG] EOS token reached at generation \(generated)")
                        fflush(stdout)
                        break
                    }
                    
                    var pieceBuffer: [CChar] = Array(repeating: 0, count: 512)
                    let pieceLen = llama_detokenize(ctx, &token, 1, &pieceBuffer, 512)
                    
                    if pieceLen > 0, let piece = String(validatingCString: &pieceBuffer) {
                        result += piece
                        
                        if generated % 50 == 0 {
                            print("[DEBUG] Generated \(generated) tokens, current result: \"\(result.prefix(50))...\"")
                            fflush(stdout)
                        }
                    }
                    
                    batch.n_tokens = 1
                    batchToken.pointee = token
                    batchPos.pointee = Int32(Int(tokenCount) + generated)
                    batchNS.pointee = 1
                    batchSeq.pointee?.pointee = 0
                    batchLogits.pointee = 0
                    
                    let ret = llama_decode(ctx, &batch)
                    if ret != 0 {
                        throw LLMError.generationFailed
                    }
                    
                    generated += 1
                }
                
                print("[DEBUG] Generation completed: \(generated) tokens, \(result.count) chars")
                print("[DEBUG] Final result: \"\(result)\"")
                fflush(stdout)
            } else {
                throw LLMError.generationFailed
            }
        }
        
        return result
    }
}

internal enum LLMError: Error {
    case notInitialized
    case modelLoadFailed
    case contextCreationFailed
    case generationFailed
}

internal struct LLMConfig {
    var modelPath: String = ""
    var threads: Int32 = 4
    var contextSize: Int32 = 2048
}
