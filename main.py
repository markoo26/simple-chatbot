import os
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PINECONE_INDEX_NAME = "krk-parking-chatbot"
DB_PATH = "parking_system.db"

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (avoid re-initialising on every import)
# ---------------------------------------------------------------------------

_pinecone_index = None
_embedding_model = None


def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    return _pinecone_index


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def search_parking_info(query: str) -> str:
    """Search the knowledge base for information about the parking facility.

    Use this for any question about: location, shuttle service, security,
    pricing tiers, booking / cancellation / modification policies, payment
    methods, check-in / check-out process, vehicle size requirements, and
    general FAQs.

    Args:
        query: Natural-language question or topic to look up.
    """
    model = _get_embedding_model()
    index = _get_pinecone_index()

    embedding = model.encode([query])[0].tolist()
    results = index.query(vector=embedding, top_k=3, include_metadata=True)

    if not results["matches"]:
        return "No relevant information found in the knowledge base."

    chunks = []
    for match in results["matches"]:
        chunks.append(
            f"[Source: {match['metadata']['source']} | Score: {match['score']:.3f}]\n"
            f"{match['metadata']['text']}"
        )
    return "\n\n---\n\n".join(chunks)


@tool
def check_parking_availability(start_date: str, end_date: str) -> str:
    """Check how many parking spots are free for a given date range and show
    example spot IDs per price tier so the user can choose one for booking.

    Args:
        start_date: Arrival date in YYYY-MM-DD format.
        end_date:   Departure date in YYYY-MM-DD format.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT p.parking_id, p.eur_price_per_day
            FROM prices p
            WHERE p.parking_id NOT IN (
                SELECT DISTINCT b.parking_id
                FROM bookings b
                WHERE b.booking_start_date < ? AND b.booking_end_date > ?
            )
            ORDER BY p.eur_price_per_day
            """,
            (end_date, start_date),
        )
        available = cursor.fetchall()
    finally:
        conn.close()

    if not available:
        return f"No parking spots are available from {start_date} to {end_date}."

    budget = [(pid, p) for pid, p in available if p <= 15]
    standard = [(pid, p) for pid, p in available if 15 < p <= 30]
    premium = [(pid, p) for pid, p in available if p > 30]

    try:
        num_days = (
            datetime.strptime(end_date, "%Y-%m-%d")
            - datetime.strptime(start_date, "%Y-%m-%d")
        ).days
    except ValueError:
        num_days = None

    lines = [
        f"Available spots from {start_date} to {end_date} — total: {len(available)}\n"
    ]
    for label, tier, tier_range in [
        ("Budget", budget, "€5-15/day"),
        ("Standard", standard, "€15-30/day"),
        ("Premium", premium, "€30-50/day"),
    ]:
        if not tier:
            continue
        pid, price = tier[0]
        total_str = f" (≈ €{round(price * num_days, 2):.2f} total)" if num_days else ""
        lines.append(
            f"  {label} ({tier_range}): {len(tier)} spot(s) available\n"
            f"    → Example: spot #{pid} at €{price:.2f}/day{total_str}"
        )

    return "\n".join(lines)


@tool
def create_booking(
    parking_id: int,
    start_date: str,
    end_date: str,
    customer_name: str,
    customer_email: str,
    customer_phone: str,
) -> str:
    """Create a parking reservation.

    Before saving, the system pauses and shows the user a full booking summary
    for confirmation. The booking is only written to the database if the user
    replies 'yes'.

    Args:
        parking_id:     ID of the parking spot to reserve (from check_parking_availability).
        start_date:     Arrival date in YYYY-MM-DD format.
        end_date:       Departure date in YYYY-MM-DD format.
        customer_name:  Full name of the customer.
        customer_email: Customer's email address.
        customer_phone: Customer's phone number.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Fetch spot price
        cursor.execute(
            "SELECT eur_price_per_day FROM prices WHERE parking_id = ?", (parking_id,)
        )
        row = cursor.fetchone()
        if not row:
            return f"Error: parking spot #{parking_id} does not exist."

        price_per_day = row[0]

        try:
            num_days = (
                datetime.strptime(end_date, "%Y-%m-%d")
                - datetime.strptime(start_date, "%Y-%m-%d")
            ).days
        except ValueError:
            return "Error: invalid date format — use YYYY-MM-DD."

        if num_days <= 0:
            return "Error: departure date must be after arrival date."

        total_price = round(price_per_day * num_days, 2)

        # Double-check availability (race condition guard)
        cursor.execute(
            """
            SELECT COUNT(*) FROM bookings
            WHERE parking_id = ? AND booking_start_date < ? AND booking_end_date > ?
            """,
            (parking_id, end_date, start_date),
        )
        if cursor.fetchone()[0] > 0:
            return (
                f"Spot #{parking_id} is no longer available for those dates. "
                "Please call check_parking_availability again to find a free spot."
            )

        # --- Human-in-the-loop confirmation ---
        confirmation = interrupt(
            {
                "type": "booking_confirmation",
                "message": (
                    f"Please confirm your reservation:\n"
                    f"  Parking spot : #{parking_id}\n"
                    f"  Arrival      : {start_date}\n"
                    f"  Departure    : {end_date} ({num_days} day(s))\n"
                    f"  Daily rate   : €{price_per_day:.2f}\n"
                    f"  Total        : €{total_price:.2f}\n"
                    f"  Name         : {customer_name}\n"
                    f"  Email        : {customer_email}\n"
                    f"  Phone        : {customer_phone}\n"
                    f"\nReply 'yes' to confirm or 'no' to cancel."
                ),
            }
        )

        if str(confirmation).strip().lower() in ("yes", "y", "confirm", "ok"):
            cursor.execute(
                """
                INSERT INTO bookings (parking_id, booking_start_date, booking_end_date, total_price)
                VALUES (?, ?, ?, ?)
                """,
                (parking_id, start_date, end_date, total_price),
            )
            booking_id = cursor.lastrowid
            conn.commit()
            return (
                f"Booking confirmed!\n"
                f"  Booking ID : {booking_id}\n"
                f"  Spot       : #{parking_id}\n"
                f"  Dates      : {start_date} → {end_date} ({num_days} day(s))\n"
                f"  Total      : €{total_price:.2f}\n"
                f"  A confirmation will be sent to {customer_email}."
            )
        else:
            return "Booking cancelled. No reservation was made."
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# LLM and bound tools
# ---------------------------------------------------------------------------

_tools = [search_parking_info, check_parking_availability, create_booking]
_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, max_retries=3)
_llm_with_tools = _llm.bind_tools(_tools)

SYSTEM_PROMPT = """You are a friendly and professional customer-service agent for \
KRK Airport Parking in Kraków, Poland.

You can help customers with three things:

1. **General information** — location, shuttle service, security, pricing tiers, \
policies, FAQs, etc.
   → Use `search_parking_info`.

2. **Availability & pricing** — how many spots are free for given dates and what \
they cost.
   → Use `check_parking_availability`.

3. **Reservations** — guide the user through the booking process, then create the \
reservation.
   → Collect step-by-step: arrival date, departure date, price preference \
(budget / standard / premium), then full name, email, and phone number.
   → Use `check_parking_availability` to confirm a spot exists and to obtain a \
valid parking_id.
   → Call `create_booking` — the system will pause and show the user a full \
summary for confirmation before saving anything.

Always use YYYY-MM-DD for dates. Be concise and helpful."""


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def chatbot_node(state: MessagesState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = _llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

_tool_node = ToolNode(_tools)

_builder = StateGraph(MessagesState)
_builder.add_node("chatbot", chatbot_node)
_builder.add_node("tools", _tool_node)
_builder.set_entry_point("chatbot")
_builder.add_conditional_edges("chatbot", tools_condition)
_builder.add_edge("tools", "chatbot")

# `graph` is the name referenced in langgraph.json
graph = _builder.compile()
