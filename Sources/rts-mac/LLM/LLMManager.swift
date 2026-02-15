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
        ctxParams.n_threads = threads
        ctxParams.n_threads_batch = threads
        
        ctxPtr = try createContext(threads: threads, contextSize: contextSize, params: &ctxParams)
        guard ctxPtr != nil else {
            llama_model_free(modelPtr)
            modelPtr = nil
            llama_backend_free()
            throw LLMError.contextCreationFailed
        }
        
        let samplerParams = llama_sampler_chain_default_params()
        samplerPtr = llama_sampler_chain_init(samplerParams)
        let seqId: Int32 = 0
        _ = llama_set_sampler(ctxPtr, seqId, samplerPtr)
        
        isInitialized = true
    }
    
    func correctText(_ text: String) async throws -> String {
        guard isInitialized, let ctx = ctxPtr else {
            print("[DEBUG] LLM not initialized, throwing error")
            fflush(stdout)
            throw LLMError.notInitialized
        }
        
        let systemPrompt = customSystemPrompt ?? LLMConstants.systemPrompt
        let userPrompt = """
あなたは以下のルールを厳密に守ってください。

[ルール]
\(systemPrompt)

[入力テキスト]
\(text)

[出力形式]
- 修正後のテキスト本文のみを返す
- 説明、注釈、見出し、引用符を付けない
"""
        let fullPrompt = userPrompt + "\n\n修正後テキスト:\n"
        
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
        
        let cleaned = result
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\0", with: "")
        
        guard !cleaned.isEmpty else {
            print("[DEBUG] Empty correction output, returning original text")
            fflush(stdout)
            return text
        }
        
        return cleaned
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
        
        if let memory = llama_get_memory(ctx) {
            llama_memory_clear(memory, true)
        }
        
        guard let model = llama_get_model(ctx), let vocab = llama_model_get_vocab(model) else {
            throw LLMError.generationFailed
        }
        let vocabSize = Int(llama_vocab_n_tokens(vocab))
        guard vocabSize > 0 else {
            throw LLMError.generationFailed
        }
        
        guard let cPrompt = strdup(prompt) else {
            throw LLMError.generationFailed
        }
        defer { free(cPrompt) }
        
        let promptLen = Int32(strlen(cPrompt))
        var tokens: [llama_token] = Array(repeating: 0, count: 4096)
        var tokenCount = llama_tokenize(
            vocab,
            cPrompt,
            promptLen,
            &tokens,
            Int32(tokens.count),
            true,
            false
        )
        if tokenCount < 0 {
            let required = Int(-tokenCount)
            tokens = Array(repeating: 0, count: required)
            tokenCount = llama_tokenize(
                vocab,
                cPrompt,
                promptLen,
                &tokens,
                Int32(tokens.count),
                true,
                false
            )
        }
        guard tokenCount > 0 else {
            throw LLMError.generationFailed
        }
        
        let eos = llama_token_eos(vocab)
        
        print("[DEBUG] Tokenizing prompt...")
        print("[DEBUG] Prompt length: \(promptLen) chars, \(tokenCount) tokens")
        fflush(stdout)
        
        let batchSize = Int32(min(tokenCount, 512))
        var batch = llama_batch_init(batchSize, 0, 1)
        defer { llama_batch_free(batch) }
        
        if let batchToken = batch.token, let batchPos = batch.pos, let batchNS = batch.n_seq_id, let batchSeq = batch.seq_id, let batchLogits = batch.logits {
            var consumed = 0
            while consumed < Int(tokenCount) {
                let chunkSize = min(Int(batchSize), Int(tokenCount) - consumed)
                batch.n_tokens = Int32(chunkSize)
                
                for j in 0..<chunkSize {
                    let idx = consumed + j
                    batchToken.advanced(by: j).pointee = tokens[idx]
                    batchPos.advanced(by: j).pointee = Int32(idx)
                    batchNS.advanced(by: j).pointee = 1
                    batchSeq.advanced(by: j).pointee?.pointee = 0
                    batchLogits.advanced(by: j).pointee = (idx == Int(tokenCount) - 1) ? 1 : 0
                }
                
                let ret = llama_decode(ctx, &batch)
                if ret != 0 {
                    throw LLMError.generationFailed
                }
                
                consumed += chunkSize
            }
            
            var generated = 0
            let maxGenerated = 2048
            
            print("[DEBUG] Starting generation, max tokens: \(maxGenerated)...")
            fflush(stdout)
            
            while generated < maxGenerated {
                var token = llama_get_sampled_token_ith(ctx, 0)
                if token == llama_token_null {
                    // Fallback: backend sampler may return null if sampler chain is empty/misconfigured.
                    token = greedySampleFromLogits(ctx: ctx, vocabSize: vocabSize)
                }
                if token < 0 || Int(token) >= vocabSize {
                    print("[DEBUG] Invalid token sampled: \(token), stopping generation")
                    fflush(stdout)
                    break
                }
                
                if token == eos {
                    print("[DEBUG] EOS token reached at generation \(generated)")
                    fflush(stdout)
                    break
                }
                
                var pieceBuffer: [CChar] = Array(repeating: 0, count: 512)
                let pieceLen = llama_detokenize(vocab, &token, 1, &pieceBuffer, 512, false, false)
                
                if pieceLen > 0 {
                    let bytes = pieceBuffer.prefix(Int(pieceLen)).map { UInt8(bitPattern: $0) }
                    let piece = String(decoding: bytes, as: UTF8.self)
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
                batchLogits.pointee = 1
                
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
        
        return result
    }
    
    private func greedySampleFromLogits(ctx: OpaquePointer, vocabSize: Int) -> llama_token {
        guard let logits = llama_get_logits(ctx) else {
            return llama_token_null
        }
        
        var bestToken: llama_token = 0
        var bestLogit = -Float.greatestFiniteMagnitude
        for i in 0..<vocabSize {
            let value = logits.advanced(by: i).pointee
            if value > bestLogit {
                bestLogit = value
                bestToken = llama_token(i)
            }
        }
        return bestToken
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
