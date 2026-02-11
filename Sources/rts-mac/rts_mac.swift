import AVFoundation
import Speech

@main
struct rts_mac {
    static func main() {
        let recognizer = SpeechRecognizer()
        recognizer.start()
    }
}

class SpeechRecognizer: NSObject {
    private var audioEngine: AVAudioEngine?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let speechRecognizer: SFSpeechRecognizer?

    override init() {
        self.speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "ja-JP"))
        super.init()
        speechRecognizer?.delegate = self
    }

    func start() {
        requestAuthorization()
    }

    private func requestAuthorization() {
        SFSpeechRecognizer.requestAuthorization { authStatus in
            DispatchQueue.main.async {
                switch authStatus {
                case .authorized:
                    print("認証に成功しました")
                    self.setupAudioSession()
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
    }

    private func setupAudioSession() {
        let audioSession = AVAudioSession.sharedInstance()

        do {
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
            startRecognition()
        } catch {
            print("Audio session error: \(error)")
            exit(1)
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

        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { result, error in
            if let result = result {
                let text = result.bestTranscription.formattedString
                print("\r\(text)", terminator: "")
                fflush(stdout)
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

        RunLoop.current.run()
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

extension SpeechRecognizer: SFSpeechRecognizerDelegate {
    func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        if available {
            print("音声認識が利用可能です")
        } else {
            print("音声認識が利用できません")
        }
    }
}
