import Foundation

@available(macOS 10.15, *)
internal struct CLIConfig {
    var llmCorrect = false
    var llmModelPath: String?
    var llmThreads: Int32 = 4
    var llmContext: Int32 = 2048
    var llmSystemPromptFile: String? = nil
}

@available(macOS 10.15, *)
internal final class CLIHandler {
    static func parseArguments(_ args: [String]) -> CLIConfig {
        var config = CLIConfig()
        var i = 1
        
        while i < args.count {
            let arg = args[i]
            
            switch arg {
            case "--llm-correct":
                config.llmCorrect = true
                i += 1
            case "--llm-model":
                i += 1
                if i < args.count {
                    config.llmModelPath = args[i]
                }
                i += 1
            case "--llm-threads":
                i += 1
                if i < args.count, let threads = Int32(args[i]) {
                    config.llmThreads = threads
                }
                i += 1
            case "--llm-context":
                i += 1
                if i < args.count, let context = Int32(args[i]) {
                    config.llmContext = context
                }
                i += 1
            case "--no-llm":
                config.llmCorrect = false
                i += 1
            case "--llm-system-prompt-file":
                i += 1
                if i < args.count {
                    config.llmSystemPromptFile = args[i]
                }
                i += 1
            default:
                i += 1
            }
        }
        
        return config
    }
}
