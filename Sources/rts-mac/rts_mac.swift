import AVFoundation
import Speech
import Foundation

@main
struct rts_mac {
    static func main() {
        setbuf(stdout, nil)
        setbuf(stderr, nil)
        
        let args = CommandLine.arguments
        
        if #available(macOS 10.15, *) {
            let config = CLIHandler.parseArguments(args)
            
            print("CLI config: llmCorrect=\(config.llmCorrect), model=\(config.llmModelPath ?? "nil")")
            fflush(stdout)
            
            let configTask = Task.detached {
                await LLMCorrectionService.shared.configure(config: config)
            }
            
            _ = Task.detached {
                await configTask.value
            }
            
            if #available(macOS 26.0, *) {
                let analyzer = SpeechAnalyzerImpl(config: config)
                analyzer.start()
            } else {
                let recognizer = SpeechRecognizerImpl(config: config)
                recognizer.start()
            }
            
            RunLoop.current.run()
        } else {
            print("このアプリはmacOS 10.15以降が必要です")
            exit(1)
        }
    }
}

@available(macOS 26.0, *)
@MainActor
class SpeechAnalyzerImpl: NSObject {
    private var audioEngine: AVAudioEngine?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let speechRecognizer: SFSpeechRecognizer?
    private let speechAnalyzer: Speech.SpeechAnalyzer?
    private let config: CLIConfig

    init(config: CLIConfig) {
        self.speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "ja-JP"))
        self.speechAnalyzer = Speech.SpeechAnalyzer(modules: [])
        self.config = config
        super.init()
        speechRecognizer?.delegate = self
    }

    func start() {
        requestAuthorization()
    }

    private func requestAuthorization() {
        SFSpeechRecognizer.requestAuthorization { authStatus in
            switch authStatus {
            case .authorized:
                print("認証に成功しました (SpeechAnalyzer)")
                self.startRecognition()
            case .denied:
                print("認証が拒否されました")
                exit(1)
            case .restricted, .notDetermined:
                print("認証できませんでした")
                exit(1)
            @unknown default:
                print("不明なエラー")
                exit(1)
            }
        }
    }

    private func startRecognition() {
        guard let speechRecognizer = speechRecognizer, speechRecognizer.isAvailable else {
            print("音声認識が利用できません")
            exit(1)
        }

        recognitionTask?.cancel()
        recognitionTask = nil

        let audioEngine = AVAudioEngine()
        self.audioEngine = audioEngine

        let inputNode = audioEngine.inputNode

        let recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        self.recognitionRequest = recognitionRequest

        recognitionRequest.shouldReportPartialResults = true
        recognitionRequest.requiresOnDeviceRecognition = true

        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { result, error in
            if let result = result {
                Task {
                    await self.handleRecognitionResult(result)
                }
            }

            if let error = error {
                print("\nError: \(error)")
                self.stopRecognition()
            }
        }

        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            self.recognitionRequest?.append(buffer)
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
            print("聞き取り中... (Ctrl+Cで終了)")
        } catch {
            print("Audio engine error: \(error)")
            exit(1)
        }
    }

    private func handleRecognitionResult(_ result: SFSpeechRecognitionResult) async {
        let text = result.bestTranscription.formattedString

        if result.isFinal {
            let corrected = await LLMCorrectionService.shared.correctText(text)
            print("\n\(corrected)")
        } else {
            print("\r\(text)", terminator: "")
        }
        fflush(stdout)
    }

    private func stopRecognition() {
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        recognitionTask?.cancel()
        recognitionTask = nil
    }
}

@available(macOS 10.15, *)
@MainActor
class SpeechRecognizerImpl: NSObject {
    private var audioEngine: AVAudioEngine?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let speechRecognizer: SFSpeechRecognizer?
    private let config: CLIConfig
    private var lastText = ""

    init(config: CLIConfig) {
        self.speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "ja-JP"))
        self.config = config
        super.init()
        speechRecognizer?.delegate = self
    }

    func start() {
        requestAuthorization()
    }

    private func requestAuthorization() {
        SFSpeechRecognizer.requestAuthorization { authStatus in
            switch authStatus {
            case .authorized:
                print("認証に成功しました (SpeechRecognizer)")
                self.startRecognition()
            case .denied:
                print("認証が拒否されました")
                exit(1)
            case .restricted, .notDetermined:
                print("認証できませんでした")
                exit(1)
            @unknown default:
                print("不明なエラー")
                exit(1)
            }
        }
    }

    private func startRecognition() {
        guard let speechRecognizer = speechRecognizer, speechRecognizer.isAvailable else {
            print("音声認識が利用できません")
            exit(1)
        }

        recognitionTask?.cancel()
        recognitionTask = nil

        let audioEngine = AVAudioEngine()
        self.audioEngine = audioEngine

        let inputNode = audioEngine.inputNode

        let recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        self.recognitionRequest = recognitionRequest

        recognitionRequest.shouldReportPartialResults = true
        recognitionRequest.requiresOnDeviceRecognition = true

        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { result, error in
            if let result = result {
                Task {
                    await self.handleRecognitionResult(result)
                }
            }

            if let error = error {
                print("\nError: \(error)")
                self.stopRecognition()
            }
        }

        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            self.recognitionRequest?.append(buffer)
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
            print("聞き取り中... (Ctrl+Cで終了)")
        } catch {
            print("Audio engine error: \(error)")
            exit(1)
        }
    }

    private func handleRecognitionResult(_ result: SFSpeechRecognitionResult) async {
        let text = result.bestTranscription.formattedString
        lastText = text

        if result.isFinal {
            let corrected = await LLMCorrectionService.shared.correctText(text)
            print("\n\(corrected)")
        } else {
            print("\r\(text)", terminator: "")
        }
        fflush(stdout)
    }

    private func stopRecognition() {
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        recognitionTask?.cancel()
        recognitionTask = nil
    }
}

@available(macOS 10.15, *)
extension SpeechRecognizerImpl: SFSpeechRecognizerDelegate {
    nonisolated func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        if available {
            print("音声認識が利用可能です")
        } else {
            print("音声認識が利用できません")
        }
    }
}

@available(macOS 26.0, *)
extension SpeechAnalyzerImpl: SFSpeechRecognizerDelegate {
    nonisolated func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        if available {
            print("音声認識が利用可能です")
        } else {
            print("音声認識が利用できません")
        }
    }
}
