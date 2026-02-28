"""
Generate evaluation data by running questions through the LangGraph chatbot.

Usage:
    python generate_eval_data.py                   # generates answers only
    python generate_eval_data.py --with-reference  # also calls GPT-4o oracle for reference_answer

Output: eval_data.json
"""

import argparse
import json

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

load_dotenv(override=True)

from main import graph  # noqa: E402 — must be after load_dotenv

# ---------------------------------------------------------------------------
# Questions to evaluate — edit this list freely
# ---------------------------------------------------------------------------

QUESTIONS = [
    "How far is the parking facility from the airport?",
    "Is there a free shuttle to the airport?",
    "What is the cancellation policy if I cancel 3 days before my arrival?",
    "What are the available parking price tiers?",
    "What is the maximum vehicle size allowed?",
    "Can I book a parking spot on the same day?",
    "How do I enter the parking facility when I arrive?",
    "Is covered parking available, and what are the benefits?",
    "What are the facility's operating hours?",
    "What happens if I stay longer than my originally booked period?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(content) -> str:
    """Handle both plain-string and list-of-blocks AIMessage content (Anthropic style)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def run_question(question: str) -> dict:
    """Invoke the graph with a single question and return an eval record."""
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    messages = result["messages"]

    # Collect context chunks returned by search_parking_info
    context_chunks = [
        msg.content
        for msg in messages
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "search_parking_info"
    ]
    context = "\n\n---\n\n".join(context_chunks)

    # Last AIMessage that is a plain text response (no pending tool calls)
    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            answer = _extract_text(msg.content)
            break

    return {
        "question": question,
        "context": context,
        "answer": answer,
        "reference_answer": "",  # filled by --with-reference or manually
    }


def generate_reference(question: str, context: str) -> str:
    """Call GPT-4o as a trusted oracle to produce a reference answer from context."""
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert on KRK Airport Parking. "
                    "Using ONLY the provided context, write a complete and accurate answer. "
                    "Do not add information not present in the context."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-reference",
        action="store_true",
        help="Generate reference_answer via GPT-4o oracle (costs tokens)",
    )
    parser.add_argument(
        "--output",
        default="eval_data.json",
        help="Output JSON file (default: eval_data.json)",
    )
    args = parser.parse_args()

    records = []
    for i, question in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {question}")
        record = run_question(question)

        if args.with_reference and record["context"]:
            print("  → generating reference answer via GPT-4o…")
            record["reference_answer"] = generate_reference(
                record["question"], record["context"]
            )

        records.append(record)
        print(f"  ✓ {record['answer'][:90]}{'…' if len(record['answer']) > 90 else ''}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(records)} records → {args.output}")
    if not args.with_reference:
        print("Tip: rerun with --with-reference to auto-fill reference_answer via GPT-4o.")


if __name__ == "__main__":
    main()
