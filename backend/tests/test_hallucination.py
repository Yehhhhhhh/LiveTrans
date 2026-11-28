import pytest

from asr import clean_text

# Real lecture speech that happens to contain words Whisper also hallucinates: it must survive.
SPEECH = [
    "Le slide sono su www.deib.polimi.it, nella sezione del corso.",
    "You can download the slides from www.polimi.it after the lecture.",
    "Amazon.com uses this kind of recommendation system.",
    "The dataset is on https://example.org/data, in the second folder.",
    "Iscriviti all'esame entro venerdì sul portale.",
    "Subscribe to the mailing list of the course to get the updates.",
    "Grazie per l'attenzione, ci vediamo giovedì per la prossima lezione.",
    "Grazie per l’attenzione, ci vediamo giovedì.",  # typographic apostrophe
    "Iscriviti al canale Teams del corso, lì trovi le registrazioni.",
]

# What Whisper writes on noise or silence: subtitle credits, video outros, a bare web address.
HALLUCINATED = [
    "Sottotitoli creati dalla comunità Amara.org",
    "Sottotitoli a cura di QTSS",
    "Grazie per la visione!",
    "Thanks for watching!",
    "Thank you for watching, and see you next time.",
    "Grazie per l'attenzione.",
    "Grazie per l’attenzione.",
    "Grazie a tutti. Grazie per l'attenzione.",
    "Iscriviti al canale!",
    "Iscriviti.",
    "Please subscribe.",
    "Subscribe to my channel.",
    "Like and subscribe!",
    "www.fema.gov",
    "https://www.example.com",
    "Grazie, grazie.",
    "Thank you.",
]


@pytest.mark.parametrize("text", SPEECH)
def test_real_speech_is_kept(text):
    assert clean_text(text) == text


@pytest.mark.parametrize("text", HALLUCINATED)
def test_hallucinations_are_dropped(text):
    assert clean_text(text) == ""


def test_repeat_loop_is_collapsed_not_dropped():
    assert clean_text("ruota, ruota, ruota, ruota del carro") == "ruota, del carro"
    assert clean_text("e e e e") == ""
