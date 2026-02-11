import Foundation

internal final class LLMBridge {
    
    internal typealias llama_free_model = @convention(c) (OpaquePointer?) -> Void
    
    internal typealias llama_free = @convention(c) (OpaquePointer?) -> Void
    
    internal typealias llama_init_from_file = @convention(c) (
        UnsafePointer<CChar>,
        Int32,
        Int32,
        Int32,
        Bool,
        Bool,
        Int32,
        Int32
    ) -> OpaquePointer?
    
    internal typealias llama_new_context_with_model = @convention(c) (
        OpaquePointer?,
        Int32,
        Int32,
        Int32,
        Int32,
        UnsafeMutablePointer<Float>?,
        Bool,
        Bool,
        Bool,
        Bool,
        Bool,
        Int32,
        Int32,
        Float,
        Float,
        Bool,
        Int32,
        Int32
    ) -> OpaquePointer?
    
    internal typealias llama_tokenize = @convention(c) (
        OpaquePointer?,
        UnsafePointer<CChar>,
        Int32,
        UnsafeMutablePointer<Int32>,
        Int32,
        Bool,
        Bool
    ) -> Int32
    
    internal typealias llama_get_model = @convention(c) (OpaquePointer?) -> OpaquePointer?
    
    internal typealias llama_n_vocab = @convention(c) (OpaquePointer?) -> Int32
    
    internal typealias llama_get_logits = @convention(c) (OpaquePointer?) -> UnsafeMutablePointer<Float>?
    
    internal typealias llama_token_to_piece = @convention(c) (
        OpaquePointer?,
        Int32,
        UnsafeMutablePointer<CChar>,
        Int32
    ) -> Int32
    
    internal typealias llama_decode = @convention(c) (
        OpaquePointer?,
        Int32,
        UnsafeMutablePointer<Int32>?,
        UnsafeMutablePointer<Float>?,
        UnsafeMutablePointer<Int32>?,
        UnsafeMutablePointer<Int32>?,
        UnsafeMutablePointer<UnsafeMutablePointer<Int32>?>?,
        UnsafeMutablePointer<Int8>?,
        Int32,
        Int32,
        Int32
    ) -> Int32
    
    internal typealias llama_sample_token_greedy = @convention(c) (
        OpaquePointer?,
        OpaquePointer?,
        UnsafePointer<Float>,
        Int32
    ) -> Int32
    
    internal typealias llama_token_bos = @convention(c) (OpaquePointer?) -> Int32
    
    internal typealias llama_token_eos = @convention(c) (OpaquePointer?) -> Int32
    
    internal typealias llama_sample_top_p_top_k = @convention(c) (
        OpaquePointer?,
        OpaquePointer?,
        UnsafeMutablePointer<Float>,
        Int32,
        Float,
        Int32,
        Float
    ) -> Void
    
    internal typealias llama_get_logits_ith = @convention(c) (
        OpaquePointer?,
        Int32
    ) -> UnsafeMutablePointer<Float>?
}
