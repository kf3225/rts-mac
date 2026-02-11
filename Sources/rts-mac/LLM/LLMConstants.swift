import Foundation

internal enum LLMConstants {
    internal static let correctionPrompt = """
    以下のテキストを修正してください。
    
    制約事項:
    - 意味を変えないでください
    - 誤字脱字のみ修正してください
    - 句読点を補完してください
    - フィラー言葉（「えー」「あのー」など）を削除してください
    - 要約や追加説明は禁止です
    - 修正後のテキストのみ出力してください
    
    テキスト:
    """
}
