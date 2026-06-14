from functools import lru_cache

from deep_translator import GoogleTranslator
from langdetect import DetectorFactory, LangDetectException, detect


DetectorFactory.seed = 0


@lru_cache(maxsize=256)
def detect_language(text):
	cleaned_text = (text or "").strip()

	if len(cleaned_text) < 3:
		return "en"

	try:
		return detect(cleaned_text)
	except LangDetectException:
		return "en"
	except Exception:
		return "en"


def translate_text(text, source_language="auto", target_language="en"):
	cleaned_text = (text or "").strip()

	if not cleaned_text:
		return ""

	if source_language == target_language:
		return cleaned_text

	try:
		return GoogleTranslator(
			source=source_language,
			target=target_language,
		).translate(cleaned_text)
	except Exception:
		return cleaned_text


def prepare_multilingual_input(text):
	language = detect_language(text)

	if language == "en":
		return {
			"language": language,
			"translated_text": text,
			"was_translated": False,
		}

	translated_text = translate_text(
		text,
		source_language=language,
		target_language="en",
	)

	return {
		"language": language,
		"translated_text": translated_text,
		"was_translated": translated_text != text,
	}


def translate_response_if_needed(text, target_language):
	if not text or target_language == "en":
		return text

	return translate_text(
		text,
		source_language="en",
		target_language=target_language,
	)
