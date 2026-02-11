# LLM校正機能実装ドキュメント

## 5. Package.swift の変更点

```swift
// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "rts-mac",
    targets: [
        .executableTarget(
            name: "rts-mac",
            cSettings: [
                .unsafeFlags(["-I/usr/local/include", "-I/opt/homebrew/include"]),
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-L/usr/local/lib", 
                    "-L/opt/homebrew/lib", 
                    "-Xlinker", "-undefined", "-Xlinker", "dynamic_lookup"
                ]),
            ]
        ),
    ]
)
```

**変更点**:
- Cヘッダー検索パスを追加 (`/usr/local/include`, `/opt/homebrew/include`)
- ライブラリ検索パスを追加 (`/usr/local/lib`, `/opt/homebrew/lib`)
- 動的リンカオプションを追加 (`-undefined dynamic_lookup`) により、llamaライブラリがない場合でもビルド可能

## 6. リアルタイム処理フロー

```
┌─────────────────┐
│  音声入力        │
│  (AVAudioEngine)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SFSpeechRecognizer│
│   (部分結果)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CLI表示         │
│ (リアルタイム)  │
└─────────────────┘

         │ result.isFinal = true
         ▼
┌─────────────────┐
│ LLMCorrection   │
│ Service         │
│ (非同期)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLMManager      │
│ (llama.cpp推論) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 校正後テキスト  │
│ 出力           │
└─────────────────┘
```

**並列処理**:
- 音声取得スレッド: 常に実行中
- LLM推論: 非同期実行（ブロックしない）
- 結果出力: 推論完了後に実行

## 7. エラーハンドリング方針

### LLM初期化エラー
```swift
do {
    try await LLMManager.shared.initialize(...)
} catch {
    print("LLM初期化失敗: \(error)")
    isEnabled = false
    // 元の音声認識機能は継続
}
```

### 推論エラー
```swift
do {
    let corrected = try await LLMManager.shared.correctText(text)
    return corrected
} catch {
    print("校正エラー: \(error)")
    return text  // エラー時は元のテキストを返す
}
```

### メモリリーク防止
```swift
func shutdown() async {
    if isLLMInitialized {
        await LLMManager.shared.shutdown()
        isLLMInitialized = false
    }
}
```

## 8. テスト戦略

### ユニットテスト

**CLIHandlerテスト**:
```swift
func testCLIArgumentsParsing() {
    let args = ["rts-mac", "--llm-correct", "--llm-model", "model.gguf", "--llm-threads", "8"]
    let config = CLIHandler.parseArguments(args)
    
    XCTAssertTrue(config.llmCorrect)
    XCTAssertEqual(config.llmModelPath, "model.gguf")
    XCTAssertEqual(config.llmThreads, 8)
}
```

**LLMCorrectionServiceテスト**:
```swift
func testLLMDisabledReturnsOriginalText() async {
    let service = LLMCorrectionService.shared
    let config = CLIConfig(llmCorrect: false)
    await service.configure(config: config)
    
    let text = "テスト"
    let result = await service.correctText(text)
    XCTAssertEqual(result, text)
}
```

### 統合テスト

**llama.cpp必須テスト**:
```swift
func testLLMCorrectionWithMockModel() async {
    let service = LLMCorrectionService.shared
    let config = CLIConfig(llmCorrect: true, llmModelPath: "/path/to/mock.gguf")
    await service.configure(config: config)
    
    let text = "あー　テストだよ"
    let result = await service.correctText(text)
    XCTAssertNotEqual(result, text)
}
```

## llama.cppインストール手順

### Homebrewを使用する場合
```bash
brew install llama.cpp
```

### ソースからビルドする場合
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
make install
```

### Metal対応ビルド
```bash
make LLAMA_METAL=1
make install
```

### GGUFモデルのダウンロード
```bash
# 例: 日本語LLMモデル
# https://huggingface.co/models?search=gguf+ja
wget https://huggingface.co/.../model.gguf
```

## 使用例

### LLM補正なし（従来挙動）
```bash
swift run
# または
swift run -- --no-llm
```

### LLM補正あり
```bash
swift run -- --llm-correct --llm-model /path/to/model.gguf --llm-threads 8
```

### 全オプション
```bash
swift run -- \
  --llm-correct \
  --llm-model ./models/japanese-7b-gguf-q4_0.gguf \
  --llm-threads 8 \
  --llm-context 4096
```

## 実装における仮定

1. **llama.cppのC API**: 現在のllama.cpp (masterブランチ) のAPIを想定
2. **Metalサポート**: macOSでのGPUアクセラレーションを前提
3. **モデル形式**: GGUF量子化モデル (Q4_0, Q5_K_M など) を想定
4. **日本語モデル**: 日本語に対応したLLMモデルが必要
5. **システムリソース**: メモリ4GB以上、CPU/GPUリソースが十分であること

## 既存機能への影響

- **変更なし**: `--no-llm` またはオプションなしの場合、従来の挙動を維持
- **後方互換**: 既存のCLIスクリプトは変更なしで動作
- **パフォーマンス**: LLM未使用時にはオーバーヘッドなし
