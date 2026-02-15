import Foundation

@_silgen_name("llama_backend_init")
internal func llama_backend_init()

@_silgen_name("llama_backend_free")
internal func llama_backend_free()

@_silgen_name("llama_model_load_from_file")
internal func llama_model_load_from_file(
    _ path: UnsafePointer<CChar>,
    _ params: UnsafePointer<llama_model_params>
) -> OpaquePointer?

@_silgen_name("llama_model_free")
internal func llama_model_free(_ model: OpaquePointer?)

@_silgen_name("llama_init_from_model")
internal func llama_init_from_model(
    _ model: OpaquePointer?,
    _ params: UnsafePointer<llama_context_params>
) -> OpaquePointer?

@_silgen_name("llama_free")
internal func llama_free(_ ctx: OpaquePointer?)

@_silgen_name("llama_model_default_params")
internal func llama_model_default_params() -> llama_model_params

@_silgen_name("llama_context_default_params")
internal func llama_context_default_params() -> llama_context_params

@_silgen_name("llama_get_model")
internal func llama_get_model(_ ctx: OpaquePointer?) -> OpaquePointer?

@_silgen_name("llama_model_get_vocab")
internal func llama_model_get_vocab(_ model: OpaquePointer?) -> OpaquePointer?

@_silgen_name("llama_vocab_n_tokens")
internal func llama_vocab_n_tokens(_ vocab: OpaquePointer?) -> Int32

@_silgen_name("llama_detokenize")
internal func llama_detokenize(
    _ vocab: OpaquePointer?,
    _ tokens: UnsafePointer<llama_token>,
    _ n_tokens: Int32,
    _ dest: UnsafeMutablePointer<CChar>?,
    _ dest_size: Int32,
    _ remove_special: Bool,
    _ unparse_special: Bool
) -> Int32

@_silgen_name("llama_tokenize")
internal func llama_tokenize(
    _ vocab: OpaquePointer?,
    _ text: UnsafePointer<CChar>,
    _ text_len: Int32,
    _ tokens: UnsafeMutablePointer<llama_token>,
    _ n_max_tokens: Int32,
    _ add_special: Bool,
    _ parse_special: Bool
) -> Int32

@_silgen_name("llama_decode")
internal func llama_decode(
    _ ctx: OpaquePointer?,
    _ batch: UnsafePointer<llama_batch>
) -> Int32

@_silgen_name("llama_batch_init")
internal func llama_batch_init(
    _ n_tokens: Int32,
    _ embd: Int32,
    _ n_seq_max: Int32
) -> llama_batch

@_silgen_name("llama_batch_free")
internal func llama_batch_free(_ batch: llama_batch)

@_silgen_name("llama_token_bos")
internal func llama_token_bos(_ vocab: OpaquePointer?) -> llama_token

@_silgen_name("llama_token_eos")
internal func llama_token_eos(_ vocab: OpaquePointer?) -> llama_token

@_silgen_name("llama_get_logits")
internal func llama_get_logits(_ ctx: OpaquePointer?) -> UnsafeMutablePointer<Float>?

@_silgen_name("llama_sampler_chain_init")
internal func llama_sampler_chain_init(
    _ params: llama_sampler_chain_params
) -> OpaquePointer?

@_silgen_name("llama_sampler_chain_default_params")
internal func llama_sampler_chain_default_params() -> llama_sampler_chain_params

@_silgen_name("llama_sampler_free")
internal func llama_sampler_free(_ sampler: OpaquePointer?)

@_silgen_name("llama_set_sampler")
internal func llama_set_sampler(
    _ ctx: OpaquePointer?,
    _ seq_id: Int32,
    _ smpl: OpaquePointer?
) -> Bool

@_silgen_name("llama_get_sampled_token_ith")
internal func llama_get_sampled_token_ith(_ ctx: OpaquePointer?, _ i: Int32) -> llama_token

@_silgen_name("llama_get_memory")
internal func llama_get_memory(_ ctx: OpaquePointer?) -> OpaquePointer?

@_silgen_name("llama_memory_clear")
internal func llama_memory_clear(_ memory: OpaquePointer?, _ clear_seq: Bool)

internal typealias llama_token = Int32
internal typealias llama_pos = Int32
internal typealias llama_seq_id = Int32
internal let llama_token_null: llama_token = -1

internal struct llama_model_params {
    var devices: UnsafeMutablePointer<OpaquePointer?>? = nil
    var tensor_buft_overrides: UnsafePointer<llama_model_tensor_buft_override>? = nil
    var n_gpu_layers: Int32 = 0
    var split_mode: Int32 = 0
    var main_gpu: Int32 = 0
    var tensor_split: UnsafePointer<Float>? = nil
    var progress_callback: (@convention(c) (Float, UnsafeMutableRawPointer?) -> Bool)? = nil
    var progress_callback_user_data: UnsafeMutableRawPointer? = nil
    var kv_overrides: UnsafePointer<llama_model_kv_override>? = nil
    var vocab_only: Bool = false
    var use_mmap: Bool = true
    var use_direct_io: Bool = false
    var use_mlock: Bool = false
    var check_tensors: Bool = true
    var use_extra_bufts: Bool = false
    var no_host: Bool = false
    var no_alloc: Bool = false
}

internal struct llama_context_params {
    var n_ctx: UInt32 = 2048
    var n_batch: UInt32 = 512
    var n_ubatch: UInt32 = 512
    var n_seq_max: UInt32 = 1
    var n_threads: Int32 = 4
    var n_threads_batch: Int32 = 4
    var rope_scaling_type: Int32 = -1
    var pooling_type: Int32 = -1
    var attention_type: Int32 = -1
    var flash_attn_type: Int32 = -1
    var rope_freq_base: Float = 0.0
    var rope_freq_scale: Float = 0.0
    var yarn_ext_factor: Float = 0.0
    var yarn_attn_factor: Float = 0.0
    var yarn_beta_fast: Float = 0.0
    var yarn_beta_slow: Float = 0.0
    var yarn_orig_ctx: UInt32 = 0
    var defrag_thold: Float = 0.0
    var cb_eval: UnsafeMutableRawPointer? = nil
    var cb_eval_user_data: UnsafeMutableRawPointer? = nil
    var type_k: Int32 = 0
    var type_v: Int32 = 0
    var abort_callback: UnsafeMutableRawPointer? = nil
    var abort_callback_data: UnsafeMutableRawPointer? = nil
    var embeddings: Bool = false
    var offload_kqv: Bool = true
    var no_perf: Bool = false
    var op_offload: Bool = false
    var swa_full: Bool = false
    var kv_unified: Bool = false
    var samplers: UnsafeMutablePointer<llama_sampler_seq_config>? = nil
    var n_samplers: Int = 0
}

internal struct llama_sampler_seq_config {
    var seq_id: llama_seq_id = 0
    var sampler: OpaquePointer? = nil
}

internal struct llama_sampler_chain_params {
    var no_perf: Bool = false
}

internal struct llama_batch {
    var n_tokens: Int32 = 0
    var token: UnsafeMutablePointer<llama_token>? = nil
    var embd: UnsafeMutablePointer<Float>? = nil
    var pos: UnsafeMutablePointer<llama_pos>? = nil
    var n_seq_id: UnsafeMutablePointer<Int32>? = nil
    var seq_id: UnsafeMutablePointer<UnsafeMutablePointer<llama_seq_id>?>? = nil
    var logits: UnsafeMutablePointer<Int8>? = nil
}

internal struct llama_model_kv_override {
    var _opaque0: Int64 = 0
    var _opaque1: Int64 = 0
    var _opaque2: Int64 = 0
    var _opaque3: Int64 = 0
    var _opaque4: Int64 = 0
    var _opaque5: Int64 = 0
    var _opaque6: Int64 = 0
    var _opaque7: Int64 = 0
    var _opaque8: Int64 = 0
    var _opaque9: Int64 = 0
    var _opaque10: Int64 = 0
    var _opaque11: Int64 = 0
    var _opaque12: Int64 = 0
    var _opaque13: Int64 = 0
    var _opaque14: Int64 = 0
    var _opaque15: Int64 = 0
    var _opaque16: Int64 = 0
    var _opaque17: Int64 = 0
    var _opaque18: Int64 = 0
    var _opaque19: Int64 = 0
    var _opaque20: Int64 = 0
}

internal struct llama_model_tensor_buft_override {
    var pattern: UnsafePointer<CChar>? = nil
    var buft: OpaquePointer? = nil
}

internal struct llama_sampling_params {
    var prev_size: Int32 = 0
    var copy_prev: Bool = false
    var n_prev: Int32 = 64
    var temperature: Float = 0.8
    var top_k: Int32 = 40
    var top_p: Float = 0.95
    var min_p: Float = 0.05
    var tfs_z: Float = 0.0
    var typical_p: Float = 0.0
    var penalty_last_n: Int32 = 64
    var penalty_repeat: Float = 0.0
    var penalty_freq: Float = 0.0
    var penalty_present: Float = 0.0
    var mirostat: Int32 = 0
    var mirostat_tau: Float = 5.0
    var mirostat_eta: Float = 0.1
    var mirostat_M: Int32 = 0
    var penalize_nl: Bool = true
    var samplers_sequence: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>? = nil
    var samplers_sequence_count: Int32 = 0
}

internal struct llama_profiling_config {
    var n_iters: Int32 = 0
    var n_outputs: Int32 = 0
    var n_print: Int32 = 0
}

internal struct llama_grammar {
    var _internal: UnsafeMutableRawPointer? = nil
}

internal struct llama_sampler {
    var _internal: UnsafeMutableRawPointer? = nil
}
