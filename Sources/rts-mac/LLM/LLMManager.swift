import Foundation

@available(macOS 10.15, *)
internal final actor LLMManager {
    static let shared = LLMManager()
    
    private var modelPtr: OpaquePointer?
    private var ctxPtr: OpaquePointer?
    private var isInitialized = false
    private let config: LLMConfig
    
    private init() {
        self.config = LLMConfig()
    }
    
    func initialize(modelPath: String, threads: Int32, contextSize: Int32) throws {
        guard !isInitialized else { return }
        
        modelPtr = try loadModel(path: modelPath)
        guard modelPtr != nil else {
            throw LLMError.modelLoadFailed
        }
        
        ctxPtr = try createContext(threads: threads, contextSize: contextSize)
        guard ctxPtr != nil else {
            throw LLMError.contextCreationFailed
        }
        
        isInitialized = true
    }
    
    func correctText(_ text: String) async throws -> String {
        guard isInitialized, let ctx = ctxPtr else {
            throw LLMError.notInitialized
        }
        
        let prompt = LLMConstants.correctionPrompt + text + "\n修正後:"
        return try await generateText(prompt: prompt, ctx: ctx)
    }
    
    func shutdown() {
        guard isInitialized else { return }
        
        if let ctx = ctxPtr {
            llama_free_impl(ctx)
            ctxPtr = nil
        }
        
        if let model = modelPtr {
            llama_free_model_impl(model)
            modelPtr = nil
        }
        
        isInitialized = false
    }
    
    private func loadModel(path: String) throws -> OpaquePointer? {
        guard let ptr = path.withCString({ cPath in
            llama_init_from_file_simple(cPath)
        }) else {
            throw LLMError.modelLoadFailed
        }
        return ptr
    }
    
    private func createContext(threads: Int32, contextSize: Int32) throws -> OpaquePointer? {
        guard let model = modelPtr else {
            throw LLMError.modelLoadFailed
        }
        
        guard let ctx = llama_new_context_with_model_simple(model, threads, contextSize) else {
            throw LLMError.contextCreationFailed
        }
        return ctx
    }
    
    private func generateText(prompt: String, ctx: OpaquePointer) async throws -> String {
        var result = ""
        
        try prompt.withCString { cPrompt in
            let promptLen = Int32(strlen(cPrompt))
            var tokens: [Int32] = Array(repeating: 0, count: 4096)
            
            let tokenCount = llama_tokenize_simple(ctx, cPrompt, promptLen, &tokens, 4096, true, false)
            
            var nPast = Int32(0)
            let eos = llama_token_eos_impl(ctx)
            
            for i in 0..<tokenCount {
                try llama_decode_simple(ctx, tokens[Int(i)], nPast)
                nPast += 1
            }
            
            var generated = 0
            while generated < 512 {
                let token = try llama_sample_token_greedy_impl(ctx, eos)
                
                if token == eos {
                    break
                }
                
                var pieceBuffer: [CChar] = Array(repeating: 0, count: 256)
                let pieceLen = llama_token_to_piece_impl(ctx, token, &pieceBuffer, 255)
                
                if pieceLen > 0, let piece = String.init(validatingCString: &pieceBuffer) {
                    result += piece
                }
                
                try llama_decode_simple(ctx, token, nPast)
                nPast += 1
                generated += 1
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

@_silgen_name("llama_init_from_file")
private func llama_init_from_file_simple(
    _ path: UnsafePointer<CChar>,
    _ n_gpu_layers: Int32 = 0,
    _ main_gpu: Int32 = 0,
    _ vocab_only: Bool = false,
    _ use_mmap: Bool = true,
    _ use_mlock: Bool = false,
    _ progress: Int32 = 0,
    _ progress_data: Int32 = 0
) -> OpaquePointer?

@_silgen_name("llama_new_context_with_model")
private func llama_new_context_with_model_simple(
    _ model: OpaquePointer,
    _ n_threads: Int32 = 4,
    _ n_batch: Int32 = 512,
    _ n_gpu_layers: Int32 = 0,
    _ main_gpu: Int32 = 0,
    _ tensor_split: UnsafeMutablePointer<Float>? = nil,
    _ f16_kv: Bool = true,
    _ logits_all: Bool = false,
    _ embedding: Bool = false,
    _ use_mmap: Bool = true,
    _ use_mlock: Bool = false,
    _ n_threads_batch: Int32 = 4,
    _ rope_freq_base: Float = 10000.0,
    _ rope_freq_scale: Float = 1.0,
    _ mul_mat_q: Bool = true,
    _ type_k: Int32 = 0,
    _ type_v: Int32 = 0
) -> OpaquePointer?

@_silgen_name("llama_free")
internal func llama_free_impl(_ ctx: OpaquePointer?)

@_silgen_name("llama_free_model")
internal func llama_free_model_impl(_ model: OpaquePointer?)

@_silgen_name("llama_tokenize")
private func llama_tokenize_simple(
    _ ctx: OpaquePointer,
    _ text: UnsafePointer<CChar>,
    _ text_len: Int32,
    _ tokens: UnsafeMutablePointer<Int32>,
    _ n_max_tokens: Int32,
    _ add_bos: Bool,
    _ special: Bool
) -> Int32

@_silgen_name("llama_decode")
private func llama_decode_simple(
    _ ctx: OpaquePointer,
    _ token: Int32,
    _ n_past: Int32
) throws

@_silgen_name("llama_sample_token_greedy")
private func llama_sample_token_greedy_impl(
    _ ctx: OpaquePointer,
    _ eos: Int32
) throws -> Int32

@_silgen_name("llama_token_to_piece")
private func llama_token_to_piece_impl(
    _ ctx: OpaquePointer,
    _ token: Int32,
    _ buf: UnsafeMutablePointer<CChar>,
    _ length: Int32
) -> Int32

@_silgen_name("llama_token_bos")
private func llama_token_bos_impl(_ ctx: OpaquePointer) -> Int32

@_silgen_name("llama_token_eos")
private func llama_token_eos_impl(_ ctx: OpaquePointer) -> Int32

@_silgen_name("llama_get_logits")
private func llama_get_logits_impl(_ ctx: OpaquePointer) -> UnsafeMutablePointer<Float>?
