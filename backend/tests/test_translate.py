from translate import build_user_message


def test_context_with_translations():
    msg = build_user_message(
        "terza frase",
        [{"src": "prima frase", "tgt": "第一句"}, {"src": "seconda frase", "tgt": "第二句"}],
        [],
    )
    assert "原文：prima frase\n译文：第一句\n原文：seconda frase\n译文：第二句" in msg
    assert msg.endswith("请翻译这一段：\nterza frase")


def test_context_still_being_translated_has_no_empty_translation_line():
    # Translations run in parallel: an earlier segment may not have its translation yet.
    msg = build_user_message(
        "terza frase",
        [{"src": "prima frase", "tgt": "第一句"}, {"src": "seconda frase", "tgt": ""}],
        [],
    )
    assert "原文：prima frase\n译文：第一句\n原文：seconda frase\n\n请翻译这一段" in msg
    assert "译文：\n" not in msg
    assert not msg.rstrip().endswith("译文：")
