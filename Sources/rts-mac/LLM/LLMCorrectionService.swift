import Foundation

@available(macOS 10.15, *)
internal actor LLMCorrectionService {
    static let shared = LLMCorrectionService()
    
    private var isEnabled = false
    private var isLLMInitialized = false
    private var customSystemPrompt: String? = nil
    
    private init() {}
    
    func configure(config: CLIConfig) async {
        isEnabled = config.llmCorrect
        print("[DEBUG] LLMCorrectionService.configure() called, isEnabled=\(isEnabled)")
        fflush(stdout)
        
        if let promptFile = config.llmSystemPromptFile {
            do {
                let fileContents = try String(contentsOfFile: promptFile, encoding: .utf8)
                customSystemPrompt = fileContents
                print("システムプロンプトをファイルから読み込みました: \(promptFile)")
                print("プロンプト（先頭100文字）: \(fileContents.prefix(100))...")
                fflush(stdout)
            } catch {
                print("エラー: システムプロンプトファイルの読み込みに失敗しました: \(error)")
                print("デフォルトのシステムプロンプトを使用します")
                fflush(stdout)
            }
        }
        
        await LLMManager.shared.setCustomSystemPrompt(customSystemPrompt)
        
        if isEnabled, let modelPath = config.llmModelPath {
            print("[DEBUG] Attempting LLM initialization with model: \(modelPath)")
            fflush(stdout)
            
            do {
                try await LLMManager.shared.initialize(
                    modelPath: modelPath,
                    threads: config.llmThreads,
                    contextSize: config.llmContext
                )
                
                isLLMInitialized = true
                print("[DEBUG] LLM initialization completed successfully")
                print("LLM初期化成功: \(modelPath)")
            } catch {
                print("[DEBUG] LLM initialization failed: \(error)")
                print("LLM初期化失敗: \(error)")
                isLLMInitialized = false
                isEnabled = false
            }
        } else {
            print("[DEBUG] LLM not enabled or no model path provided")
            fflush(stdout)
        }
        
        print("[DEBUG] LLMCorrectionService.configure() completed")
        print("[DEBUG] isEnabled=\(isEnabled), isLLMInitialized=\(isLLMInitialized)")
        fflush(stdout)
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
