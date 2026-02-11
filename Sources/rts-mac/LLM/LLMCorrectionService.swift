import Foundation

@available(macOS 10.15, *)
internal actor LLMCorrectionService {
    static let shared = LLMCorrectionService()
    
    private var isEnabled = false
    private var isLLMInitialized = false
    
    private init() {}
    
    func configure(config: CLIConfig) {
        isEnabled = config.llmCorrect
        
        if isEnabled, let modelPath = config.llmModelPath {
            Task {
                do {
                    try await LLMManager.shared.initialize(
                        modelPath: modelPath,
                        threads: config.llmThreads,
                        contextSize: config.llmContext
                    )
                    isLLMInitialized = true
                    print("LLM初期化成功: \(modelPath)")
                } catch {
                    print("LLM初期化失敗: \(error)")
                    isEnabled = false
                }
            }
        }
    }
    
    func correctText(_ text: String) async -> String {
        guard isEnabled, isLLMInitialized else {
            return text
        }
        
        do {
            let corrected = try await LLMManager.shared.correctText(text)
            return corrected
        } catch {
            print("校正エラー: \(error)")
            return text
        }
    }
    
    func shutdown() async {
        if isLLMInitialized {
            await LLMManager.shared.shutdown()
            isLLMInitialized = false
        }
    }
}
