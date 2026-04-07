"""
tests/test_enrichment.py — Full pipeline test (all 5 agents).

Usage:
    python -m tests.test_enrichment
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def test_enrichment_agent():
    from src.graph import run_pipeline

    print("\n" + "="*60)
    print("  Content Intelligence Agent — Full Pipeline Test")
    print("="*60 + "\n")

    result = run_pipeline(
        title="Turkish for Beginners",
        synopsis=(
            "A German comedy series about a patchwork family where "
            "a German girl's mother marries a Turkish man, leading to "
            "cultural clashes and comedic situations."
        ),
    )

    print(f"Run ID       : {result['run_id']}")
    print(f"Completed    : {result['completed_agents']}")
    print(f"Errors       : {result['errors'] or 'none'}")

    # --- Enrichment ---
    print("\n--- Enrichment ---")
    enrichment = result.get("enrichment")
    if enrichment:
        print(f"Confidence   : {enrichment['confidence']:.0%}")
        print(f"Snippets     : {len(enrichment['web_snippets'])} fetched")
        for key, value in enrichment["facts"].items():
            print(f"  {key:<22} {value}")
    else:
        print("  Failed or skipped.")

    # --- Analysis ---
    print("\n--- Analysis ---")
    analysis = result.get("analysis")
    if analysis:
        print(f"Confidence   : {analysis['confidence']:.0%}")
        print(f"Genre        : {analysis['genre_tags']}")
        print(f"Mood         : {analysis['mood_tags']}")
        print(f"Audience     : {analysis['audience_tags']}")
        print(f"Warnings     : {analysis['content_warnings']}")
    else:
        print("  Failed or skipped.")

    # --- Sentiment ---
    print("\n--- Sentiment ---")
    sentiment = result.get("sentiment")
    if sentiment:
        print(f"Confidence   : {sentiment['confidence']:.0%}")
        print(f"Score        : {sentiment['score']}/100")
        print(f"Sources      : {sentiment['source_count']}")
        print(f"Summary      : {sentiment['summary']}")
        print(f"Positives    : {sentiment['positives']}")
        print(f"Criticisms   : {sentiment['criticisms']}")
    else:
        print("  Failed or skipped.")

    # --- Generation ---
    print("\n--- Generation ---")
    generation = result.get("generation")
    if generation:
        print(f"Confidence   : {generation['confidence']:.0%}")
        print(f"\nSEO title    : {generation['seo_title']}")
        print(f"\nMeta desc    : {generation['meta_description']}")
        print(f"  ({len(generation['meta_description'])} chars)")
        print(f"\nInstagram    :\n{generation['instagram_post']}")
        print(f"\nTwitter/X    :\n{generation['twitter_post']}")
        print(f"  ({len(generation['twitter_post'])} chars)")
    else:
        print("  Failed or skipped.")

    # --- Evaluation ---
    print("\n--- Evaluation ---")
    evaluation = result.get("evaluation")
    if evaluation:
        print(f"Overall conf : {evaluation['overall_confidence']:.0%}")
        print(f"Halluc flags : {evaluation['hallucination_flags'] or 'none'}")
        print(f"Low conf     : {evaluation['low_confidence_agents'] or 'none'}")
        print(f"Est. tokens  : {evaluation['total_tokens_used']}")
    else:
        print("  Failed or skipped.")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    test_enrichment_agent()