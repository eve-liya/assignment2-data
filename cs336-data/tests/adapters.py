#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any
from cs336_data import extract_text, identify_text, deduplication, quality_classifier


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text.extract_text(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return identify_text.identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return identify_text.mask_email(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return identify_text.mask_phone_num(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return identify_text.mask_ip(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return identify_text.identify_nsfw(text)

def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return identify_text.identify_hatespeech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    return quality_classifier.load_and_predict(text)

def run_gopher_quality_filter(text: str) -> bool:
    return quality_classifier.gopher_quality_filters(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    deduplication.exact_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    deduplication.run_minhash_deduplication(input_files, num_hashes, num_bands, ngrams, output_directory, jaccard_threshold)
